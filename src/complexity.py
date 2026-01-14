"""Complexity metrics module.

Compute complexity metrics such as spectral entropy and dominant frequency ratio.
"""

import numpy as np
from scipy.signal import welch

from .utils import detrend_series


def spectral_entropy(values):
    """Compute spectral predictability (1 - normalized spectral entropy).

    Uses LOESS detrending to remove trend, then computes the power spectral
    density using Welch's method (averaged periodograms with overlapping segments).
    Entropy is normalized and converted to predictability in [0, 1].
    Returns NaN for degenerate inputs.
    
    Args:
        values: array-like, time series values
        
    Returns:
        float: spectral predictability in [0, 1]
    """
    y = np.asarray(values, dtype=float)
    n = y.size
    if n < 2 or not np.isfinite(y).all():
        return np.nan

    # LOESS detrending to remove trend
    y_detrended = detrend_series(y)

    # Compute power spectral density using Welch's method
    # This reduces noise by averaging periodograms of overlapping segments
    # and automatically applies windowing (default: Hann window)
    _, psd = welch(y_detrended, nperseg=min(256, n))

    # Normalize to create probability distribution
    power_sum = psd.sum()
    
    p = psd / power_sum
    eps = np.finfo(float).eps  # small epsilon to prevent taking log(0)
    
    # Spectral entropy using natural logarithm
    entropy = -np.sum(p * np.log(p + eps))
    
    # Maximum entropy for a uniform distribution over m frequency bins is log(m)
    m = len(psd)
    max_entropy = np.log(m)

    complexity = entropy / max_entropy
    
    return complexity