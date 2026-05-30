from dataclasses import dataclass
from typing import Optional

@dataclass
class SWEAConfig:
    """Sliding Window Ensembleの設定"""
    
    sink_size: int = 10
    window_size: int = 512
    local_size: int = 100
    ensemble_n: float = 1.5
    min_tokens: int = field(init=False)
# __post_init__ で:
self.min_tokens = self.sink_size + self.window_size + self.local_size  # sink + window + local の合計

    def __post_init__(self):
        if self.window_size % 2 != 0:
            self.window_size = (self.window_size // 2) * 2  # 偶数に調整
