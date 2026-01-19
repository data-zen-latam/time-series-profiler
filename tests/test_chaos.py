"""Unit tests for chaos metrics.

Demonstrates pytest features:
- @pytest.mark.parametrize for data-driven tests
- pytest.approx for float comparisons with tolerance
- fixtures for shared test data
- pytest.raises for exception testing
"""

import numpy as np
import pytest

from src.chaos import (_estimate_delay_autocorrelation, cao_e1_e2,
                       estimate_embedding_dimension_cao,
                       largest_lyapunov_exponent)

# Fixtures are defined in conftest.py and automatically available


class TestEstimateDelayAutocorrelation:
    """Test delay estimation via autocorrelation."""
    
    def test_constant_series_returns_one(self, constant_series):
        """Constant series should return delay of 1 (guard case)."""
        delay = _estimate_delay_autocorrelation(constant_series)
        assert delay == 1
        
    def test_periodic_series_finds_delay(self, sine_series):
        """Periodic series should find a reasonable delay."""
        delay = _estimate_delay_autocorrelation(sine_series)
        assert delay >= 1
        assert delay < len(sine_series) // 2
    
    # Parametrized test: Run same test with different inputs
    @pytest.mark.parametrize("series_length", [50, 100, 500, 1000])
    def test_returns_integer_for_various_lengths(self, series_length):
        """Delay should always be integer, regardless of series length."""
        series = np.random.randn(series_length)
        delay = _estimate_delay_autocorrelation(series)
        assert isinstance(delay, (int, np.integer))


class TestCaoE1E2:
    """Test Cao's E1 and E2 statistics."""
    
    def test_constant_series_returns_nan(self, constant_series):
        """Constant series should return NaN arrays (guard case)."""
        e1, e2 = cao_e1_e2(constant_series, max_dim=5, tau=1)
        assert np.all(np.isnan(e1))
        assert np.all(np.isnan(e2))
    
    # Parametrized test: Verify shapes for different max_dim values
    @pytest.mark.parametrize("max_dim", [3, 5, 8, 12])
    def test_returns_correct_shapes(self, white_noise, max_dim):
        """E1 and E2 should have shape (max_dim-1,)."""
        e1, e2 = cao_e1_e2(white_noise, max_dim=max_dim, tau=1)
        assert e1.shape == (max_dim - 1,)
        assert e2.shape == (max_dim - 1,)
        
    def test_e1_values_positive(self):
        """E1 values should be non-negative for valid series."""
        t = np.linspace(0, 20*np.pi, 500)
        series = np.sin(t) + 0.1 * np.random.randn(500)
        e1, e2 = cao_e1_e2(series, max_dim=5, tau=10)
        assert np.all(e1[~np.isnan(e1)] >= 0)


class TestEstimateEmbeddingDimensionCao:
    """Test Cao's embedding dimension estimation."""
    
    def test_constant_series_returns_minimum(self, constant_series):
        """Constant series should return minimum dimension (2)."""
        m = estimate_embedding_dimension_cao(constant_series, max_dim=10, tau=1)
        assert m == 2
    
    # Parametrized test: Verify output is always in valid range
    @pytest.mark.parametrize("max_dim", [5, 8, 10, 15])
    def test_returns_integer_in_range(self, white_noise, max_dim):
        """Dimension should be integer between 2 and max_dim."""
        m = estimate_embedding_dimension_cao(white_noise, max_dim=max_dim, tau=1)
        assert isinstance(m, (int, np.integer))
        assert 2 <= m <= max_dim
        
    def test_periodic_series_low_dimension(self, sine_series):
        """Simple periodic series should find low embedding dimension."""
        m = estimate_embedding_dimension_cao(sine_series, max_dim=10, tau=10)
        assert m <= 5  # Simple sine should need low dimension


class TestLargestLyapunovExponent:
    """Test largest Lyapunov exponent computation."""
    
    # Parametrized test: Multiple constant values should all give λ=0
    @pytest.mark.parametrize("value", [0.0, 1.0, 42.0, -5.0, 1e6])
    def test_constant_series_returns_zero(self, value):
        """Any constant series should have λ = 0.0 (no divergence)."""
        series = np.full(500, value)
        lyap = largest_lyapunov_exponent(series)
        assert lyap == pytest.approx(0.0, abs=1e-10)
        
    def test_pure_sine_near_zero(self, sine_series):
        """Pure sine wave should have λ ≈ 0 (not chaotic)."""
        lyap = largest_lyapunov_exponent(sine_series)
        # Using pytest.approx with tolerance
        assert lyap == pytest.approx(0.0, abs=0.5)
        
    def test_lorenz_positive(self, lorenz_series):
        """Lorenz attractor should have positive λ (chaotic)."""
        lyap = largest_lyapunov_exponent(lorenz_series)
        assert lyap > 0.0  # Lorenz should be positive (chaotic)
        
    def test_white_noise_behavior(self, white_noise):
        """White noise should return a valid value (not NaN)."""
        lyap = largest_lyapunov_exponent(white_noise)
        assert not np.isnan(lyap)
        assert not np.isinf(lyap)
    
    def test_series_with_valid_values_only(self):
        """Series with only valid values should return valid result."""
        # NaN/inf handling is edge case; ensure valid data works
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 100)
        lyap = largest_lyapunov_exponent(series)
        assert not np.isnan(lyap)
        assert not np.isinf(lyap)
    
    def test_reasonable_length_series(self):
        """Series of reasonable length should return valid value."""
        # Very short series may return NaN; test with sufficient data
        series = np.arange(100, dtype=float)
        lyap = largest_lyapunov_exponent(series)
        assert not np.isinf(lyap)
        
    def test_returns_float(self, white_noise):
        """Lyapunov exponent should return float."""
        lyap = largest_lyapunov_exponent(white_noise)
        assert isinstance(lyap, (float, np.floating))
    
    # Parametrized test: Different parameter combinations
    @pytest.mark.parametrize("embedding_dim,delay,evolution_steps", [
        (3, 5, 10),
        (4, 10, 20),
        (2, 1, 5),
    ])
    def test_explicit_parameters(self, sine_series, embedding_dim, delay, evolution_steps):
        """Should work with explicitly provided parameters."""
        lyap = largest_lyapunov_exponent(
            sine_series,
            embedding_dim=embedding_dim,
            delay=delay,
            evolution_steps=evolution_steps
        )
        assert not np.isnan(lyap)
        assert isinstance(lyap, (float, np.floating))
    
    def test_deterministic_with_fixed_series(self):
        """Same input should produce same output (deterministic)."""
        series = np.sin(np.linspace(0, 20*np.pi, 500))
        lyap1 = largest_lyapunov_exponent(series)
        lyap2 = largest_lyapunov_exponent(series)
        assert lyap1 == lyap2
