"""Utility functions for SWEA."""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from transformers import PreTrainedTokenizer


def show_top5(
    logits_or_probs: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    label: str = "Top-5 Predictions",
) -> None:
    """Display top-5 predictions from logits or probabilities.
    
    Args:
        logits_or_probs (torch.Tensor): Logits or probability tensor.
        tokenizer (PreTrainedTokenizer): Tokenizer for decoding token IDs.
        label (str): Label for output.
    """
    # Detect if input is logits or probabilities
    if logits_or_probs.min().item() < 0 or logits_or_probs.sum().item() > 1.5:
        probs = F.softmax(logits_or_probs, dim=-1)
    else:
        probs = logits_or_probs  # Already normalized
    
    top5 = torch.topk(probs, 5)
    print(f"\n{label}:")
    for prob, idx in zip(top5.values[0], top5.indices[0]):
        token = tokenizer.decode(idx)
        print(f"  '{token}' : {prob.item():.4f}")


def count_windows(
    total_len: int,
    sink_size: int,
    window_size: int,
    local_size: int,
) -> int:
    """Count the number of windows for a given sequence length.
    
    Args:
        total_len (int): Total sequence length.
        sink_size (int): Sink region size.
        window_size (int): Window size.
        local_size (int): Local region size.
    
    Returns:
        int: Number of windows.
    """
    stride = window_size // 2
    mid_end = total_len - local_size
    windows = 0
    pos = sink_size
    
    while pos + window_size <= mid_end:
        windows += 1
        pos += stride
    
    last_start = mid_end - window_size
    if last_start >= sink_size and (windows == 0 or pos - stride != last_start):
        windows += 1
    
    return windows
