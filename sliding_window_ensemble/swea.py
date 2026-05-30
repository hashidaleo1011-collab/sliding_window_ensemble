"""Sliding Window Ensemble with KV-Cache (Batched Version)"""

import torch
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional, Any
from transformers import PreTrainedModel, PreTrainedTokenizer

from .config import SWEAConfig


class SWEAWithCache:
    """Sliding Window Ensemble with KV-Cache (バッチ処理対応版)"""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: Optional[SWEAConfig] = None,
        sink_size: Optional[int] = None,
        window_size: Optional[int] = None,
        local_size: Optional[int] = None,
        ensemble_n: Optional[float] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        
        # Config設定
        if config is None:
            config = SWEAConfig()
        self.config = config

        # パラメータ上書き
        if sink_size is not None:
            self.config.sink_size = sink_size
        if window_size is not None:
            self.config.window_size = window_size
        if local_size is not None:
            self.config.local_size = local_size
        if ensemble_n is not None:
            self.config.ensemble_n = ensemble_n

        # KVキャッシュ
        self.kv_cache: Dict[Tuple[int, int, int], Any] = {}

    def predict_next_logits(
        self,
        input_ids: torch.Tensor,   # (batch_size, seq_len)
        n: Optional[float] = None,
    ) -> Tuple[torch.Tensor, str]:
        """
        次のトークンの確率分布を予測（バッチ処理対応）
        """
        _n = n if n is not None else self.config.ensemble_n
        batch_size, total_len = input_ids.shape

        # 短いシーケンスの場合は通常処理
        if total_len < self.config.min_tokens:
            with torch.no_grad():
                out = self.model(input_ids)
            return out.logits[:, -1, :], "standard_generation"

        mid_end = total_len - self.config.local_size
        windows = self._build_windows(total_len)
        
        if not windows:
            return self._fallback(input_ids)

        # ==================== バッチ化処理 ====================
        local_ids = input_ids[:, mid_end:]                    # (batch, local_size)
        num_windows = len(windows)

        # 1. 各ウィンドウのKVキャッシュを準備
        past_list = []
        for w_start, w_end in windows:
            key = (w_start, w_end, mid_end)
            
            if key not in self.kv_cache:
                # SINK + WINDOW のキャッシュ作成
                sink_part = input_ids[:, :self.config.sink_size]
                window_part = input_ids[:, w_start:min(w_end, mid_end)]
                prefix_ids = torch.cat([sink_part, window_part], dim=1)
                
                with torch.no_grad():
                    prefix_out = self.model(prefix_ids, use_cache=True)
                self.kv_cache[key] = prefix_out.past_key_values
            
            past_list.append(self.kv_cache[key])

        # 2. KVキャッシュをバッチ化
        batched_past = self._stack_past_key_values(past_list)

        # 3. LOCAL部分を全ウィンドウ分まとめて処理（ここが最大の高速化ポイント）
        batched_local = local_ids.repeat(num_windows, 1)   # (num_windows * batch, local_size)

        with torch.no_grad():
            out = self.model(
                batched_local,
                past_key_values=batched_past,
                use_cache=False,
            )

        logits = out.logits[:, -1, :]                      # (num_windows * batch, vocab_size)
        probs = F.softmax(logits, dim=-1)

        # 4. アンサンブル融合
        probs = probs.view(num_windows, batch_size, -1)    # (num_windows, batch, vocab)
        fused_probs = (probs ** _n).mean(dim=0)            # (batch, vocab)
        fused_probs = fused_probs / fused_probs.sum(dim=-1, keepdim=True)

        mode_desc = f"batched_ensemble(w={num_windows},n={_n})"
        return fused_probs, mode_desc

    def _stack_past_key_values(self, past_list: List) -> Tuple:
        """複数のpast_key_valuesをバッチ次元で結合"""
        if not past_list:
            return None

        layers = len(past_list[0])
        batched_past = []

        for layer_idx in range(layers):
            keys = [past[layer_idx][0] for past in past_list]
            values = [past[layer_idx][1] for past in past_list]
            
            batched_key = torch.cat(keys, dim=0)
            batched_value = torch.cat(values, dim=0)
            
            batched_past.append((batched_key, batched_value))

        return tuple(batched_past)

    def _build_windows(self, total_len: int) -> List[Tuple[int, int]]:
        """ウィンドウ位置の作成（元のロジックを維持）"""
        stride = self.config.window_size // 2
        mid_end = total_len - self.config.local_size
        windows = []
        pos = self.config.sink_size

        while pos + self.config.window_size <= mid_end:
            windows.append((pos, pos + self.config.window_size))
            pos += stride

        # 最後のウィンドウを必ず追加
        last_start = max(self.config.sink_size, mid_end - self.config.window_size)
        if last_start >= self.config.sink_size:
            if not windows or windows[-1][0] != last_start:
                windows.append((last_start, mid_end))

        return windows

    def _fallback(self, input_ids: torch.Tensor):
        """フォールバック処理"""
        with torch.no_grad():
            out = self.model(input_ids)
        return out.logits[:, -1, :], "fallback"

    def clear_cache(self) -> None:
        """新しい生成を開始する前に呼んでください"""
        self.kv_cache.clear()
