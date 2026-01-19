"""Shared pytest fixtures for all test modules.

This file is automatically discovered by pytest and makes fixtures
available to all test files in the tests/ directory.
"""

import numpy as np
import pytest

# === Basic Series Fixtures ===

@pytest.fixture
def constant_series():
    """Constant series for testing zero-variance guards."""
    return np.ones(500)


@pytest.fixture
def sine_series():
    """Pure sine wave for testing periodic behavior."""
    t = np.linspace(0, 50*np.pi, 1000)
    return np.sin(t)


@pytest.fixture
def multi_frequency():
    """Multiple frequencies (moderate complexity)."""
    t = np.linspace(0, 20*np.pi, 1000)
    return np.sin(t) + 0.5*np.sin(3*t) + 0.3*np.sin(5*t)


@pytest.fixture
def white_noise():
    """White noise for testing high-complexity behavior."""
    np.random.seed(42)
    return np.random.randn(1000)


@pytest.fixture
def lorenz_series():
    """Lorenz attractor x-component (chaotic).
    
    Expensive to generate, so defined as fixture to reuse across tests.
    """
    def lorenz(x, y, z, s=10, r=28, b=2.667):
        dx = s * (y - x)
        dy = x * (r - z) - y
        dz = x * y - b * z
        return dx, dy, dz
    
    dt = 0.01
    x, y, z = 1.0, 1.0, 1.0
    xs = []
    for _ in range(5000):
        dx, dy, dz = lorenz(x, y, z)
        x += dx * dt
        y += dy * dt
        z += dz * dt
        xs.append(x)
    return np.array(xs)


# === Data Quality Specific Fixtures ===

@pytest.fixture
def perfect_series():
    """Perfect quality series: no missing, no outliers, all unique."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def normal_distribution():
    """Normal distribution for outlier testing."""
    np.random.seed(42)
    return np.random.randn(1000)


@pytest.fixture
def series_with_outliers():
    """Series with clear outliers."""
    return np.array([1, 2, 3, 4, 5, 100, 200])
