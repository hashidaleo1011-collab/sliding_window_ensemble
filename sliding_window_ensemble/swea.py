"""Sliding Window Ensemble with KV-Cache (Batched Version)"""

import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple

from transformers import PreTrainedModel, PreTrainedTokenizer

from .config import SWEAConfig


class SWEAWithCache:
    """Sliding Window Ensemble with KV-Cache（バッチ処理対応版）

    長いシーケンスを SINK / WINDOW / LOCAL の 3 領域に分割し、
    複数のウィンドウ位置で予測した確率をN乗融合することで
    推論精度を保ちながら KV キャッシュ再利用によるコスト削減を実現する。

    Args:
        model:       HuggingFace CausalLM モデル
        tokenizer:   対応するトークナイザー
        config:      SWEAConfig インスタンス（省略時はデフォルト値）
        sink_size:   config を上書きする場合に指定
        window_size: config を上書きする場合に指定
        local_size:  config を上書きする場合に指定
        ensemble_n:  config を上書きする場合に指定
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: Optional[SWEAConfig] = None,
        sink_size: Optional[int] = None,
        window_size: Optional[int] = None,
        local_size: Optional[int] = None,
        ensemble_n: Optional[float] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer

        # Config を構築（キーワード引数で直接渡された値を優先）
        if config is None:
            config = SWEAConfig(
                sink_size=sink_size if sink_size is not None else 10,
                window_size=window_size if window_size is not None else 512,
                local_size=local_size if local_size is not None else 100,
                ensemble_n=ensemble_n if ensemble_n is not None else 1.5,
            )
        else:
            # 既存 config に個別上書き → _recalc() で min_tokens を再計算
            if sink_size is not None:
                config.sink_size = sink_size
            if window_size is not None:
                config.window_size = window_size
            if local_size is not None:
                config.local_size = local_size
            if ensemble_n is not None:
                config.ensemble_n = ensemble_n
            config._recalc()

        self.config = config

        # KV キャッシュ: key=(w_start, w_end, mid_end, content_hash) → past_key_values
        self.kv_cache: Dict[Tuple[int, int, int, int], Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_next_logits(
        self,
        input_ids: torch.Tensor,  # (batch_size, seq_len)
        n: Optional[float] = None,
    ) -> Tuple[torch.Tensor, str]:
        """次のトークンの確率分布を返す（バッチ対応）

        Args:
            input_ids: 入力トークン列。shape = (batch_size, seq_len)
            n:         アンサンブルのN乗係数（省略時は config.ensemble_n）

        Returns:
            (fused_probs, mode_str)
            fused_probs: shape = (batch_size, vocab_size)
            mode_str:    動作モードの説明文字列
        """
        _n = n if n is not None else self.config.ensemble_n
        batch_size, total_len = input_ids.shape

        # シーケンスが短い場合は通常の forward で処理
        if total_len < self.config.min_tokens:
            with torch.no_grad():
                out = self.model(input_ids)
            probs = F.softmax(out.logits[:, -1, :], dim=-1)
            return probs, "standard_generation"

        mid_end = total_len - self.config.local_size
        windows = self._build_windows(total_len)

        if not windows:
            return self._fallback(input_ids)

        # ---- 1. 各ウィンドウの KV キャッシュを準備 ----
        local_ids = input_ids[:, mid_end:]  # (batch, local_size)
        past_list: List[Any] = []

        for w_start, w_end in windows:
            sink_part = input_ids[:, : self.config.sink_size]
            window_part = input_ids[:, w_start : min(w_end, mid_end)]
            prefix_ids = torch.cat([sink_part, window_part], dim=1)

            # トークン内容をハッシュ化してキーに含める
            # → 異なるプロンプトが同じ位置でキャッシュに誤ヒットするのを防ぐ
            content_hash = hash(prefix_ids.cpu().numpy().tobytes())
            key = (w_start, w_end, mid_end, content_hash)

            if key not in self.kv_cache:
                with torch.no_grad():
                    prefix_out = self.model(prefix_ids, use_cache=True)
                self.kv_cache[key] = prefix_out.past_key_values

            past_list.append(self.kv_cache[key])

        # ---- 2. KV キャッシュをバッチ次元で結合 ----
        num_windows = len(windows)
        batched_past = self._stack_past_key_values(past_list, batch_size)

        # ---- 3. LOCAL 部分を全ウィンドウ分まとめて一括 forward ----
        # (num_windows * batch_size, local_size)
        batched_local = local_ids.repeat(num_windows, 1)

        with torch.no_grad():
            out = self.model(
                batched_local,
                past_key_values=batched_past,
                use_cache=False,
            )

        # ---- 4. N 乗アンサンブル融合 ----
        logits = out.logits[:, -1, :]                        # (num_windows * batch, vocab)
        probs = F.softmax(logits, dim=-1)
        probs = probs.view(num_windows, batch_size, -1)      # (num_windows, batch, vocab)
        fused_probs = (probs ** _n).mean(dim=0)              # (batch, vocab)
        fused_probs = fused_probs / fused_probs.sum(dim=-1, keepdim=True)

        return fused_probs, f"batched_ensemble(w={num_windows},n={_n})"

    def clear_cache(self) -> None:
        """KV キャッシュを全消去する。新しい生成を開始する前に呼ぶこと。"""
        self.kv_cache.clear()

    def generate_stream(
        self,
        text: str,
        max_new_tokens: int = 100,
        n=None,
    ):
        """1トークンできるたびに yield するストリーミング生成。

        Args:
            text:           入力テキスト
            max_new_tokens: 最大生成トークン数
            n:              アンサンブルのN乗係数（省略時は config.ensemble_n）

        Yields:
            生成されたトークンの文字列（1トークンずつ）

        使い方:
            for token in swea.generate_stream("こんにちは"):
                print(token, end="", flush=True)
        """
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(
            self.model.device
        )
        generated = input_ids.clone()
        self.clear_cache()

        for _ in range(max_new_tokens):
            probs, _ = self.predict_next_logits(generated, n=n)
            next_token_id = probs.argmax(dim=-1, keepdim=True)

            if next_token_id.item() == self.tokenizer.eos_token_id:
                break

            generated = torch.cat([generated, next_token_id], dim=1)
            token_text = self.tokenizer.decode(
                next_token_id[0], skip_special_tokens=True
            )
            yield token_text
            
    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_windows(self, total_len: int) -> List[Tuple[int, int]]:
        """ウィンドウ位置のリストを構築する。

        stride = window_size // 2 の 50% オーバーラップで配置し、
        末尾には mid_end に揃えた最終ウィンドウを必ず追加する。
        """
        stride = self.config.window_size // 2
        mid_end = total_len - self.config.local_size
        windows: List[Tuple[int, int]] = []
        pos = self.config.sink_size

        while pos + self.config.window_size <= mid_end:
            windows.append((pos, pos + self.config.window_size))
            pos += stride

        # 末尾ウィンドウ（mid_end に右端を揃える）
        last_start = max(self.config.sink_size, mid_end - self.config.window_size)
        if not windows or windows[-1][0] != last_start:
            windows.append((last_start, mid_end))

        return windows

    def _stack_past_key_values(
        self, past_list: List[Any], batch_size: int
    ) -> Any:
        """複数の past_key_values をバッチ次元で結合する。

        transformers v4（タプル形式）と v5（DynamicCache形式）の両方に対応。
        キャッシュは batch_size=1 で作成されているため、
        各ウィンドウのキャッシュを batch_size 分 repeat してから結合する。
        """
        if not past_list:
            return None

        # transformers v5: DynamicCache オブジェクトかどうか判定
        try:
            from transformers.cache_utils import DynamicCache
            is_dynamic_cache = isinstance(past_list[0], DynamicCache)
        except ImportError:
            is_dynamic_cache = False

        if is_dynamic_cache:
            # v5: DynamicCache の batch_repeat_interleave を使って結合
            import copy
            merged = DynamicCache()
            for past in past_list:
                cache_copy = copy.deepcopy(past)
                cache_copy.batch_repeat_interleave(batch_size)
                # 各ウィンドウのキャッシュを merged に追加
                for layer_idx in range(len(cache_copy.layers)):
                    layer = cache_copy.layers[layer_idx]
                    merged.update(layer.keys, layer.values, layer_idx)
            return merged

        else:
            # v4: タプル形式
            num_layers = len(past_list[0])
            batched_past = []
            for layer_idx in range(num_layers):
                keys = [
                    past[layer_idx][0].repeat(batch_size, 1, 1, 1) for past in past_list
                ]
                values = [
                    past[layer_idx][1].repeat(batch_size, 1, 1, 1) for past in past_list
                ]
                batched_past.append(
                    (torch.cat(keys, dim=0), torch.cat(values, dim=0))
                )
            return tuple(batched_past)

    def _fallback(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """ウィンドウが作れない場合の通常 forward フォールバック"""
        with torch.no_grad():
            out = self.model(input_ids)
        probs = F.softmax(out.logits[:, -1, :], dim=-1)
        return probs, "fallback"
