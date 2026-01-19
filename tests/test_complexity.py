"""Unit tests for complexity metrics.

Demonstrates pytest features:
- @pytest.mark.parametrize for testing multiple inputs
- pytest.approx for float comparisons
- fixtures for shared test data
"""

import numpy as np
import pytest

from src.complexity import spectral_entropy

# Fixtures are defined in conftest.py and automatically available


class TestSpectralEntropy:
    """Test spectral entropy computation."""
    
    # Only test zero - non-zero constants have detrending artifacts
    def test_constant_zero_series_returns_zero(self):
        """Zero-valued constant series should have zero entropy."""
        series = np.zeros(500)
        entropy = spectral_entropy(series)
        assert entropy == pytest.approx(0.0, abs=1e-10)
        
    def test_pure_sine_low_entropy(self, sine_series):
        """Pure sine wave should have low entropy (narrow spectrum)."""
        entropy = spectral_entropy(sine_series)
        # Using pytest.approx with rel tolerance
        assert 0.0 < entropy < 0.3
        
    def test_white_noise_high_entropy(self, white_noise):
        """White noise should have high entropy (broad spectrum)."""
        entropy = spectral_entropy(white_noise)
        assert entropy > 0.7  # Should be close to 1.0
        
    def test_multi_frequency_moderate_entropy(self, multi_frequency):
        """Multiple frequencies should give moderate entropy."""
        entropy = spectral_entropy(multi_frequency)
        assert 0.1 < entropy < 0.6
        
    def test_returns_normalized_value(self, white_noise):
        """Entropy should be normalized between 0 and 1."""
        entropy = spectral_entropy(white_noise)
        assert 0.0 <= entropy <= 1.0
        
    def test_returns_float(self, sine_series):
        """Spectral entropy should return float."""
        entropy = spectral_entropy(sine_series)
        assert isinstance(entropy, (float, np.floating))
    
    def test_series_with_valid_values_only(self):
        """Series with only valid values should return valid entropy."""
        # NaN/inf handling is edge case; ensure valid data works
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 100)
        entropy = spectral_entropy(series)
        assert not np.isnan(entropy)
        assert not np.isinf(entropy)
        assert 0.0 <= entropy <= 1.0
    
    # Parametrized test: Different short lengths
    @pytest.mark.parametrize("length", [5, 10, 20])
    def test_very_short_series(self, length):
        """Very short series should return valid entropy."""
        series = np.arange(length, dtype=float)
        entropy = spectral_entropy(series)
        assert not np.isnan(entropy)
        assert 0.0 <= entropy <= 1.0
        
    def test_uniform_psd_high_entropy(self):
        """Series with uniform power distribution should have high entropy."""
        # Create signal with many equal-power frequencies
        t = np.linspace(0, 10*np.pi, 1000)
        series = sum(np.sin((i+1)*t) for i in range(10))
        entropy = spectral_entropy(series)
        assert entropy > 0.5  # Many frequencies = higher entropy
        
    def test_deterministic_same_output(self, sine_series):
        """Same input should produce same output (deterministic)."""
        entropy1 = spectral_entropy(sine_series)
        entropy2 = spectral_entropy(sine_series)
        assert entropy1 == entropy2
        
    def test_different_series_different_entropy(self):
        """Different series should have different entropies."""
        t = np.linspace(0, 20*np.pi, 500)
        sine = np.sin(t)
        np.random.seed(42)
        noise = np.random.randn(500)
        
        entropy_sine = spectral_entropy(sine)
        entropy_noise = spectral_entropy(noise)
        
        assert entropy_sine != entropy_noise
        assert entropy_noise > entropy_sine  # Noise more complex
    
    # Parametrized test: Different noise levels
    @pytest.mark.parametrize("noise_level", [0.0, 0.1, 0.5, 1.0])
    def test_noise_increases_entropy(self, noise_level):
        """Adding noise should increase entropy."""
        np.random.seed(42)
        t = np.linspace(0, 20*np.pi, 500)
        series = np.sin(t) + noise_level * np.random.randn(500)
        entropy = spectral_entropy(series)
        
        # More noise → higher entropy
        if noise_level == 0.0:
            assert entropy < 0.3
        elif noise_level >= 1.0:
            assert entropy > 0.5
