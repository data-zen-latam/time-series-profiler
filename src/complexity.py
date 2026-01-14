"""Complexity metrics module.

Compute complexity metrics such as spectral entropy and dominant frequency ratio.
"""

import numpy as np
from scipy import signal
from .utils import detrend_series


def spectral_entropy(values):
    """Compute spectral predictability (1 - normalized spectral entropy).

    Uses STL decomposition to remove trend and seasonality, then applies a Hann
    window to the residuals and computes the one-sided power spectrum via FFT.
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

    # STL decomposition to extract residuals (detrended + deseasonalized)
    y_detrended = detrend_series(y)

    # Apply Hann window to reduce spectral leakage
    window = np.hanning(n)
    yw = y_detrended * window

    # One-sided power spectrum. For real-valued time series, the negative frequencies are mirror images of the positive ones and carry no additional information.
    fft_vals = np.fft.rfft(yw)
    power = np.abs(fft_vals) ** 2
    power_sum = power.sum()

    p = power / power_sum
    eps = np.finfo(float).eps  # small epsilon to prevent taking log(0).
    # Spectral entropy using natural logarithm
    entropy = -np.sum(p * np.log(p + eps))
    norm = np.log(2 * np.pi)

    predictability = 1.0 - entropy / norm
    # Clamp to [0, 1] for numerical stability
    return float(np.clip(predictability, 0.0, 1.0))
