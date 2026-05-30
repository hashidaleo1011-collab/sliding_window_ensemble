"""Tests for SWEAConfig."""

import pytest
from sliding_window_ensemble.config import SWEAConfig


class TestSWEAConfig:
    """Test SWEAConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SWEAConfig()
        assert config.sink_size == 10
        assert config.window_size == 512
        assert config.local_size == 100
        assert config.ensemble_n == 1.5

    def test_min_tokens(self):
        """Test min_tokens calculation."""
        config = SWEAConfig(sink_size=10, window_size=512, local_size=100)
        assert config.min_tokens == 622

    def test_custom_config(self):
        """Test custom configuration."""
        config = SWEAConfig(
            sink_size=5,
            window_size=256,
            local_size=50,
            ensemble_n=2.0,
        )
        assert config.sink_size == 5
        assert config.window_size == 256
        assert config.local_size == 50
        assert config.ensemble_n == 2.0

    def test_invalid_sink_size(self):
        """Test validation of sink_size."""
        with pytest.raises(ValueError):
            SWEAConfig(sink_size=0)
        with pytest.raises(ValueError):
            SWEAConfig(sink_size=-1)

    def test_invalid_window_size(self):
        """Test validation of window_size."""
        with pytest.raises(ValueError):
            SWEAConfig(window_size=0)
        with pytest.raises(ValueError):
            SWEAConfig(window_size=-1)

    def test_invalid_local_size(self):
        """Test validation of local_size."""
        with pytest.raises(ValueError):
            SWEAConfig(local_size=0)
        with pytest.raises(ValueError):
            SWEAConfig(local_size=-1)

    def test_invalid_ensemble_n(self):
        """Test validation of ensemble_n."""
        with pytest.raises(ValueError):
            SWEAConfig(ensemble_n=0)
        with pytest.raises(ValueError):
            SWEAConfig(ensemble_n=-1)
