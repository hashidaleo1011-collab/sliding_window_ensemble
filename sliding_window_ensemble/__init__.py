"""Sliding Window Ensemble (SWEA) for LLM Inference Optimization."""

from .swea import SWEAWithCache
from .config import SWEAConfig

__version__ = "0.1.0"
__author__ = "hashidaleo1011-collab"
__all__ = ["SWEAWithCache", "SWEAConfig"]
