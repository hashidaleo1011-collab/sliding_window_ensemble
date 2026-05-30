from dataclasses import dataclass, field

@dataclass
class SWEAConfig:
    sink_size: int = 10
    window_size: int = 512
    local_size: int = 100
    ensemble_n: float = 1.5
    min_tokens: int = field(init=False)

    def __post_init__(self) -> None:
        if self.sink_size <= 0:
            raise ValueError(f"sink_size は正の整数 (got {self.sink_size})")
        if self.window_size <= 0:
            raise ValueError(f"window_size は正の整数 (got {self.window_size})")
        if self.local_size <= 0:
            raise ValueError(f"local_size は正の整数 (got {self.local_size})")
        if self.ensemble_n <= 0:
            raise ValueError(f"ensemble_n は正の数 (got {self.ensemble_n})")
        if self.window_size % 2 != 0:
            self.window_size = (self.window_size // 2) * 2
        self.min_tokens = self.sink_size + self.window_size + self.local_size

    def _recalc(self) -> None:
        if self.window_size % 2 != 0:
            self.window_size = (self.window_size // 2) * 2
        self.min_tokens = self.sink_size + self.window_size + self.local_size
