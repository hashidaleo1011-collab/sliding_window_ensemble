"""Configuration for Sliding Window Ensemble."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SWEAConfig:
    """Configuration for SWEAWithCache.
    
    Attributes:
        sink_size (int): Number of tokens at the beginning (context start).
        window_size (int): Size of the sliding window.
        local_size (int): Number of recent tokens at the end.
        ensemble_n (float): Exponent for N-power ensemble fusion.
                           1.0 = normal average, larger = emphasize confidence.
    """

    sink_size: int = 10
    window_size: int = 512
    local_size: int = 100
    ensemble_n: float = 1.5

    @property
    def min_tokens(self) -> int:
        """Minimum number of tokens to use ensemble method."""
        return self.sink_size + self.window_size + self.local_size

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.sink_size <= 0:
            raise ValueError(f"sink_size must be positive, got {self.sink_size}")
        if self.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {self.window_size}")
        if self.local_size <= 0:
            raise ValueError(f"local_size must be positive, got {self.local_size}")
        if self.ensemble_n <= 0:
            raise ValueError(f"ensemble_n must be positive, got {self.ensemble_n}")
