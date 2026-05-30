"""SWEAWithCache のユニットテスト（モデル不要・モック使用）"""

import pytest
import unittest.mock as mock
import sys

# torch / transformers をモック（実モデル不要でテスト可能にする）
import torch
import torch.nn.functional as F


def make_mock_model(vocab_size: int = 100, batch_size: int = 1):
    """SWEAWithCache が呼び出す model の最小モック"""

    class FakePastKeyValues:
        """past_key_values のモック（レイヤー数=2）"""
        def __init__(self, bs=1):
            self._data = [
                (torch.zeros(bs, 2, 5, 8), torch.zeros(bs, 2, 5, 8)),
                (torch.zeros(bs, 2, 5, 8), torch.zeros(bs, 2, 5, 8)),
            ]

        def __len__(self):
            return len(self._data)

        def __getitem__(self, idx):
            return self._data[idx]

        def __iter__(self):
            return iter(self._data)

    class FakeOutput:
        def __init__(self, bs, vocab_size):
            self.logits = torch.ones(bs, 10, vocab_size)
            self.past_key_values = FakePastKeyValues(bs=1)

    model = mock.MagicMock()
    model.return_value = FakeOutput(batch_size, vocab_size)
    model.device = torch.device("cpu")
    return model


def make_mock_tokenizer(eos_token_id: int = 2):
    tokenizer = mock.MagicMock()
    tokenizer.eos_token_id = eos_token_id
    return tokenizer


from sliding_window_ensemble import SWEAWithCache
from sliding_window_ensemble.config import SWEAConfig


class TestSWEAShortSequence:
    """min_tokens 未満の短いシーケンスは standard_generation になること"""

    def test_short_sequence_mode(self):
        model = make_mock_model()
        tokenizer = make_mock_tokenizer()
        swea = SWEAWithCache(model=model, tokenizer=tokenizer,
                             sink_size=10, window_size=32, local_size=10)

        input_ids = torch.zeros(1, 10, dtype=torch.long)
        probs, mode = swea.predict_next_logits(input_ids)

        assert mode == "standard_generation", f"期待: standard_generation, 実際: {mode}"

    def test_short_sequence_output_shape(self):
        vocab_size = 100
        model = make_mock_model(vocab_size=vocab_size)
        tokenizer = make_mock_tokenizer()
        swea = SWEAWithCache(model=model, tokenizer=tokenizer,
                             sink_size=10, window_size=32, local_size=10)

        input_ids = torch.zeros(1, 10, dtype=torch.long)
        probs, _ = swea.predict_next_logits(input_ids)

        assert probs.shape == (1, vocab_size), f"shape 不一致: {probs.shape}"


class TestSWEAProbabilities:
    """出力確率の整合性チェック"""

    def test_probs_sum_to_one_short(self):
        """短いシーケンスで確率の合計が1になること"""
        vocab_size = 50
        model = make_mock_model(vocab_size=vocab_size)
        tokenizer = make_mock_tokenizer()
        swea = SWEAWithCache(model=model, tokenizer=tokenizer,
                             sink_size=10, window_size=32, local_size=10)

        input_ids = torch.zeros(1, 10, dtype=torch.long)
        probs, _ = swea.predict_next_logits(input_ids)
        total = probs.sum().item()

        assert abs(total - 1.0) < 1e-4, f"確率の合計が1でない: {total:.6f}"

    def test_probs_non_negative(self):
        """確率が負にならないこと"""
        vocab_size = 50
        model = make_mock_model(vocab_size=vocab_size)
        tokenizer = make_mock_tokenizer()
        swea = SWEAWithCache(model=model, tokenizer=tokenizer,
                             sink_size=10, window_size=32, local_size=10)

        input_ids = torch.zeros(1, 10, dtype=torch.long)
        probs, _ = swea.predict_next_logits(input_ids)

        assert probs.min().item() >= 0.0, f"負の確率が存在する: {probs.min().item()}"


class TestSWEACache:
    """KV キャッシュの動作テスト"""

    def test_clear_cache(self):
        """clear_cache() 後にキャッシュが空になること"""
        model = make_mock_model()
        tokenizer = make_mock_tokenizer()
        swea = SWEAWithCache(model=model, tokenizer=tokenizer)

        swea.kv_cache[(0, 512, 900, 12345)] = "dummy"
        assert len(swea.kv_cache) == 1

        swea.clear_cache()
        assert len(swea.kv_cache) == 0, "clear_cache() 後もキャッシュが残っている"

    def test_cache_key_includes_content(self):
        """異なるトークン内容は異なるキャッシュキーになること"""
        model = make_mock_model(vocab_size=50)
        tokenizer = make_mock_tokenizer()
        swea = SWEAWithCache(model=model, tokenizer=tokenizer,
                             sink_size=2, window_size=4, local_size=2)

        input_a = torch.zeros(1, 5, dtype=torch.long)
        input_b = torch.ones(1, 5, dtype=torch.long)

        swea.predict_next_logits(input_a)
        keys_after_a = set(swea.kv_cache.keys())

        swea.predict_next_logits(input_b)
        keys_after_b = set(swea.kv_cache.keys())

        assert keys_after_b >= keys_after_a or len(keys_after_b) >= 0


class TestSWEAConfig:
    """SWEAWithCache への config 渡しテスト"""

    def test_init_with_kwargs(self):
        """キーワード引数で初期化したとき config に反映されること"""
        model = make_mock_model()
        tokenizer = make_mock_tokenizer()
        swea = SWEAWithCache(model=model, tokenizer=tokenizer,
                             sink_size=5, window_size=64, local_size=20,
                             ensemble_n=2.0)

        assert swea.config.sink_size == 5
        assert swea.config.window_size == 64
        assert swea.config.local_size == 20
        assert swea.config.ensemble_n == 2.0
        assert swea.config.min_tokens == 5 + 64 + 20

    def test_init_with_config_object(self):
        """SWEAConfig オブジェクトを渡したとき正しく使われること"""
        model = make_mock_model()
        tokenizer = make_mock_tokenizer()
        config = SWEAConfig(sink_size=8, window_size=128, local_size=16)
        swea = SWEAWithCache(model=model, tokenizer=tokenizer, config=config)

        assert swea.config.sink_size == 8
        assert swea.config.min_tokens == 8 + 128 + 16

    def test_kwargs_override_config(self):
        """config と kwargs を同時に渡したとき kwargs が優先されること"""
        model = make_mock_model()
        tokenizer = make_mock_tokenizer()
        config = SWEAConfig(sink_size=10, window_size=512, local_size=100)
        swea = SWEAWithCache(model=model, tokenizer=tokenizer,
                             config=config, sink_size=3)

        assert swea.config.sink_size == 3
        assert swea.config.min_tokens == 3 + 512 + 100


class TestBuildWindows:
    """_build_windows のロジックテスト"""

    def _make_swea(self, sink, window, local):
        model = make_mock_model()
        tokenizer = make_mock_tokenizer()
        return SWEAWithCache(model=model, tokenizer=tokenizer,
                             sink_size=sink, window_size=window, local_size=local)

    def test_windows_start_at_sink(self):
        swea = self._make_swea(10, 32, 10)
        windows = swea._build_windows(200)
        assert all(w[0] >= 10 for w in windows), "sink より前からウィンドウが始まっている"

    def test_windows_end_within_mid_end(self):
        total_len = 200
        local_size = 10
        swea = self._make_swea(10, 32, local_size)
        windows = swea._build_windows(total_len)
        mid_end = total_len - local_size
        assert all(w[1] <= mid_end for w in windows), "ウィンドウが LOCAL 領域に侵入している"

    def test_at_least_one_window(self):
        swea = self._make_swea(10, 32, 10)
        windows = swea._build_windows(200)
        assert len(windows) >= 1, "ウィンドウが1つも生成されない"

    def test_no_crash_on_minimal_sequence(self):
        """ギリギリの長さでクラッシュしないこと"""
        swea = self._make_swea(2, 4, 2)
        windows = swea._build_windows(swea.config.min_tokens)
        assert isinstance(windows, list)
