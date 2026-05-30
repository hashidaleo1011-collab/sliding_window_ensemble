"""Sliding Window Ensemble with KV-Cache Implementation."""

import torch
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional, Any
from transformers import PreTrainedModel, PreTrainedTokenizer

from .config import SWEAConfig


class SWEAWithCache:
    """Sliding Window Ensemble (SWEA) with KV-Cache optimization.
    
    Algorithm overview:
        Each generation step divides the input token sequence into 3 regions:
            [SINK (first 10)] [WINDOW (middle 512)] [LOCAL (last 100)]
        
        - SINK: Context start (unchanged)
        - WINDOW: Sliding window (computed at multiple positions)
        - LOCAL: Most recent info (recomputed every step)
        
        Optimizations:
        1. SINK+WINDOW KV-Cache is reused when unchanged
        2. Multiple window positions ensemble predictions
        3. N-power fusion emphasizes confidence
    
    Args:
        model (PreTrainedModel): Language model for inference.
        tokenizer (PreTrainedTokenizer): Tokenizer for the model.
        config (SWEAConfig, optional): Configuration. Uses defaults if None.
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
    ):
        """Initialize SWEA.
        
        Args:
            model: PreTrainedModel instance.
            tokenizer: PreTrainedTokenizer instance.
            config: SWEAConfig instance. If None, uses defaults.
            sink_size: Override config.sink_size.
            window_size: Override config.window_size.
            local_size: Override config.local_size.
            ensemble_n: Override config.ensemble_n.
        """
        self.model = model
        self.tokenizer = tokenizer
        
        # Initialize or override config
        if config is None:
            config = SWEAConfig()
        self.config = config
        
        # Override specific parameters if provided
        if sink_size is not None:
            self.config.sink_size = sink_size
        if window_size is not None:
            self.config.window_size = window_size
        if local_size is not None:
            self.config.local_size = local_size
        if ensemble_n is not None:
            self.config.ensemble_n = ensemble_n
        
        # KV-Cache storage: (w_start, w_end, mid_end) -> past_key_values
        self.kv_cache: Dict[Tuple[int, int, int], Any] = {}

    def predict_next_logits(
        self,
        input_ids: torch.Tensor,
        n: Optional[float] = None,
    ) -> Tuple[torch.Tensor, str]:
        """Predict logits for the next token.
        
        Args:
            input_ids (torch.Tensor): Input token IDs, shape (batch_size, seq_len).
            n (float, optional): Override ensemble_n parameter.
        
        Returns:
            tuple: (logits/probs, mode_description)
                - logits/probs: shape (batch_size, vocab_size)
                - mode_description: string describing the prediction mode
        """
        _n = n if n is not None else self.config.ensemble_n
        total_len = input_ids.shape[1]
        mid_end = total_len - self.config.local_size

        # Standard generation for short sequences
        if total_len < self.config.min_tokens:
            with torch.no_grad():
                out = self.model(input_ids)
            return out.logits[:, -1, :], "standard_generation"

        # N-power ensemble + KV-Cache
        windows = self._build_windows(total_len)
        all_probs = []

        for w_start, w_end in windows:
            logits = self._predict_window(input_ids, w_start, w_end, mid_end)
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs ** _n)

        fused = torch.cat(all_probs, dim=0).mean(dim=0, keepdim=True)
        fused = fused / fused.sum(dim=-1, keepdim=True)
        
        mode_desc = f"ensemble_cache(windows={len(windows)},n={_n})"
        return fused, mode_desc

    def clear_cache(self) -> None:
        """Clear KV-Cache. Call this at the start of a new generation."""
        self.kv_cache.clear()

    def _build_windows(self, total_len: int) -> List[Tuple[int, int]]:
        """Build window positions with stride = window_size // 2.
        
        Args:
            total_len (int): Total length of input sequence.
        
        Returns:
            list: List of (w_start, w_end) tuples.
        """
        stride = self.config.window_size // 2
        mid_end = total_len - self.config.local_size
        windows = []
        pos = self.config.sink_size
        
        while pos + self.config.window_size <= mid_end:
            windows.append((pos, pos + self.config.window_size))
            pos += stride
        
        last_start = mid_end - self.config.window_size
        last_end = mid_end
        
        if last_start >= self.config.sink_size:
            if len(windows) == 0 or windows[-1] != (last_start, last_end):
                windows.append((last_start, last_end))
        
        return windows

    def _predict_window(
        self,
        input_ids: torch.Tensor,
        w_start: int,
        w_end: int,
        mid_end: int,
    ) -> torch.Tensor:
        """Predict logits for a single window.
        
        Reuses KV-Cache for SINK+WINDOW, computes LOCAL every step.
        Cache key includes mid_end to prevent gaps.
        
        Args:
            input_ids (torch.Tensor): Full input token IDs.
            w_start (int): Window start position.
            w_end (int): Window end position.
            mid_end (int): End of middle region (= total_len - local_size).
        
        Returns:
            torch.Tensor: Logits for next token, shape (batch_size, vocab_size).
        """
        window_key = (w_start, w_end, mid_end)

        if window_key not in self.kv_cache:
            # Compute SINK + WINDOW and cache KV
            actual_w_end = min(w_end, mid_end)
            sink_window_ids = torch.cat([
                input_ids[:, :self.config.sink_size],
                input_ids[:, w_start:actual_w_end]
            ], dim=1)
            
            with torch.no_grad():
                out_sw = self.model(sink_window_ids, use_cache=True)
            self.kv_cache[window_key] = out_sw.past_key_values

        # Compute LOCAL using cached KV
        local_ids = input_ids[:, mid_end:]
        with torch.no_grad():
            out = self.model(
                local_ids,
                past_key_values=self.kv_cache[window_key],
                use_cache=False,
            )
        return out.logits[:, -1, :]
