"""Chaotic behavior metrics module.

Compute chaotic behavior metrics such as the largest Lyapunov exponent via delay embedding.
Provides a normalized chaos score via a sigmoid transform for visualization.
"""

import logging

import numpy as np
from statsmodels.tsa.stattools import acf

from .utils import detrend_series

logger = logging.getLogger(__name__)

MIN_STATES_FOR_LYAPUNOV = 10


def chaos_sigmoid(x, steepness=2.0):
    """Map Lyapunov exponent values to [0, 1] via a sigmoid.

    Uses an increasing logistic function centered at x=0:
        s(x) = 1 / (1 + exp(-k * x))

    where `k` (steepness) controls the slope near x=0. A larger `k`
    yields a sharper transition around zero, helping radar charts
    separate non-chaotic (x <= 0) from chaotic (x > 0) series.

    Args:
        x: float or array-like, Lyapunov exponent(s)
        steepness: float, k parameter for sigmoid (default: 2.0)

    Returns:
        float or ndarray: values in [0, 1]
    """
    x = np.asarray(x, dtype=float)
    # Guard for non-finite values
    if not np.isfinite(x).all():
        return np.where(np.isfinite(x), 1.0 / (1.0 + np.exp(-steepness * x)), np.nan)
    return 1.0 / (1.0 + np.exp(-steepness * x))


def chaos_score(values, embedding_dim=None, delay=None, evolution_steps=20, steepness=2.0):
    """Compute a normalized chaos score from the largest Lyapunov exponent.

    Wraps `largest_lyapunov_exponent` and applies a sigmoid transform
    to yield a value in [0, 1] suitable for radar charts and comparisons.

    The default `steepness=2.0` increases slope near x==0 so that
    chaotic series (positive exponents) map close to 1 quickly,
    while non-chaotic (zero/negative) map closer to 0.

    Args:
        values: array-like, time series values
        embedding_dim: int or None, embedding dimension m (default: auto via Cao)
        delay: int or None, time delay τ for embedding (default: auto via ACF)
        evolution_steps: int, steps Δt to track divergence (default: 20)
        steepness: float, k parameter for sigmoid (default: 2.0)

    Returns:
        float: chaos score in [0, 1]; NaN if exponent cannot be estimated.
    """
    lam = largest_lyapunov_exponent(values, embedding_dim=embedding_dim, delay=delay, evolution_steps=evolution_steps)
    if np.isnan(lam):
        return np.nan
    return float(chaos_sigmoid(lam, steepness=steepness))


def _estimate_delay_autocorrelation(y, threshold=0.3):
    """Estimate optimal delay using autocorrelation method.
    
    Finds the first lag where autocorrelation drops below threshold or reaches 
    its first minimum. Assumes input is already detrended if needed.
    
    Args:
        y: array, time series values (should be detrended)
        threshold: float, ACF threshold for delay selection
        
    Returns:
        int: estimated delay τ
    """
    # Guard against constant/zero-variance series
    if np.std(y) == 0:
        return 1
    
    n = len(y)
    max_lag = min(100, n // 4)  # Heuristic. Don't search beyond 1/4 of series length.
    
    if max_lag < 2:
        return 1
    
    # Compute ACF using statsmodels (efficient implementation)
    # Returns ndarray of shape (nlags+1,) when alpha=None and qstat=False
    acf_vals = acf(y, nlags=max_lag)
    
    # Find first lag where ACF < threshold
    below_threshold = np.where(acf_vals[1:] < threshold)[0] #type: ignore
    if len(below_threshold) > 0:
        return below_threshold[0] + 1
    
    # Otherwise find first local minimum
    for lag in range(2, len(acf_vals) - 1):
        if acf_vals[lag] < acf_vals[lag - 1] and acf_vals[lag] < acf_vals[lag + 1]:
            return lag
    
    # Fallback to lag 1
    logger.warning("Could not estimate delay from ACF; defaulting to 1.")
    return 1


def cao_e1_e2(time_series, max_dim=9, tau=None):
    """Compute Cao's E1 and E2 statistics for optimal embedding dimension selection.
    
    E1(m) identifies the minimum embedding dimension where reconstruction is faithful.
    E2(m) uses random neighbors as a sanity check (should stay roughly constant).
    
    Reference: Cao, L. (1997). "Practical method for determining the minimum embedding 
    dimension of a scalar time series." Physica D, 110(1–2), 43–50.
    
    Args:
        time_series: array-like, time series values
        max_dim: int, maximum embedding dimension to test (default: 9)
        tau: int or None, time delay for embedding (default: None, auto-estimated via ACF)
        
    Returns:
        tuple: (E1 array, E2 array) of shape (max_dim-1,) where index i corresponds to ratio at m=i+1
    """
    time_series = np.asarray(time_series, dtype=float)

    if tau is None:
        tau = max(1, _estimate_delay_autocorrelation(time_series))

    # Guard against constant/zero-variance series
    if np.std(time_series) == 0:
        return np.full(max_dim - 1, np.nan), np.full(max_dim - 1, np.nan)

    N = len(time_series)
    E1_raw = []
    E2_raw = []

    for m in range(1, max_dim + 1):
        M = N - m * tau
        if M <= 1:
            break

        # Phase-space reconstruction (M x m)
        idx = np.arange(M)[:, None] + np.arange(m) * tau
        Y = time_series[idx]

        # Chebyshev (max) norm distances
        dist = np.abs(Y[:, None, :] - Y[None, :, :]).max(axis=2)
        np.fill_diagonal(dist, np.inf)

        # Nearest neighbor (exclude self)
        nn_idx = np.argmin(dist, axis=1)
        nearest_neighbor_distance = dist[np.arange(M), nn_idx]

        # Extend by next coordinate to m+1 dimensions
        next_idx = np.arange(M) + m * tau
        point_extended = np.column_stack((Y, time_series[next_idx]))
        neighbor_extended = np.column_stack((Y[nn_idx], time_series[nn_idx + m * tau]))

        dist_extended = np.abs(point_extended - neighbor_extended).max(axis=1)

        epsilon = np.finfo(float).eps
        a = dist_extended / np.maximum(nearest_neighbor_distance, epsilon)
        d = np.abs(time_series[next_idx] - time_series[nn_idx + m * tau])

        E1_raw.append(a.mean())
        E2_raw.append(d.mean())

    E1_raw = np.asarray(E1_raw, dtype=float)
    E2_raw = np.asarray(E2_raw, dtype=float)

    if E1_raw.size >= 2:
        E1 = E1_raw[1:] / E1_raw[:-1]
    else:
        E1 = np.array([])

    if E2_raw.size >= 2:
        E2 = E2_raw[1:] / E2_raw[:-1]
    else:
        E2 = np.array([])

    return E1, E2


def estimate_embedding_dimension_cao(y, max_dim=9, tau=None, e1_threshold=0.05):
    """Estimate optimal embedding dimension using Cao's method with auto-detected elbow.
    
    Finds the first dimension where the E1 ratio stabilizes near 1 (plateau),
    indicating sufficient embedding. Uses a relative tolerance around 1 instead
    of a loose upper bound to avoid collapsing to m=2.
    
    Args:
        y: array-like, time series values
        max_dim: int, maximum dimension to test (default: 10)
        tau: int or None, time delay (default: None, auto-estimated via ACF)
        e1_threshold: float, tolerance for |E1-1| to detect plateau (default: 0.05)
        
    Returns:
        int: estimated optimal embedding dimension (minimum 2)
    """
    y = np.asarray(y, dtype=float)
    
    if tau is None:
        tau = max(1, _estimate_delay_autocorrelation(y))
    
    E1, _ = cao_e1_e2(y, max_dim=max_dim, tau=tau)
    
    if E1.size == 0 or np.all(np.isnan(E1)):
        return 2

    optimal_dim = 2
    # Look for two consecutive ratios close to 1 within tolerance
    for i in range(E1.size - 1):
        if np.isfinite(E1[i]) and np.isfinite(E1[i + 1]):
            if abs(E1[i] - 1.0) < e1_threshold and abs(E1[i + 1] - 1.0) < e1_threshold:
                optimal_dim = i + 2  # m = i + 2
                break

    # Fallback: choose dimension whose ratio is closest to 1
    if optimal_dim == 2:
        finite_mask = np.isfinite(E1)
        if np.any(finite_mask):
            closest_idx = int(np.argmin(np.abs(E1[finite_mask] - 1.0)))
            # Map back to original indices
            original_indices = np.nonzero(finite_mask)[0]
            optimal_dim = int(original_indices[closest_idx]) + 2

    logger.info(f"Cao's method estimated optimal embedding dimension: {optimal_dim} (tau={tau})")
    return optimal_dim


def largest_lyapunov_exponent(values, embedding_dim=None, delay=None, evolution_steps=20):
    """Compute the largest Lyapunov exponent using Rosenstein's method.
    
    Implements Rosenstein et al.'s approach: tracks divergence of nearby trajectories
    over multiple time steps, fits a linear slope to the log-divergence curve, and
    returns the largest exponent from the m-dimensional Lyapunov spectrum.
    
    Reference: Rosenstein, M. T., Collins, J. J., & De Luca, C. J. (1993). 
    "A practical method for calculating largest Lyapunov exponents from small data sets."
    Physica D, 65(1-2), 117-134.
    
    Args:
        values: array-like, time series values
        embedding_dim: int or None, embedding dimension m (default: auto via Cao)
        delay: int or None, time delay τ for embedding (default: auto via ACF)
        evolution_steps: int or None, steps Δt to track divergence 
                         (default: max(30, 1% of series length))
    
    Returns:
        float: largest Lyapunov exponent. NaN if insufficient data.
    """
    y = np.asarray(values, dtype=float)
    y_detrended = detrend_series(y)

    n = y_detrended.size

    if not np.isfinite(y_detrended).all():
        return np.nan

    # Guard against constant/near-constant series (numerical tolerance)
    if np.std(y) == 0 or np.std(y_detrended) < 1e-9:
        return 0.0

    # Auto-estimate embedding dimension using Cao's method if not specified
    if embedding_dim is None:
        embedding_dim = estimate_embedding_dimension_cao(y_detrended, max_dim=9, tau=delay)

    # Validate input
    if n < 100 * embedding_dim:
        return np.nan  # Insufficient data for reliable estimation

    # Auto-estimate delay if not provided
    if delay is None:
        delay = _estimate_delay_autocorrelation(y_detrended)

    lambda_dims = []

    for m in range(1, embedding_dim + 1):
        # Setup for dimension m
        embedding_window = (m - 1) * delay

        # Clamp evolution_steps per dimension
        max_evolution_steps = n - embedding_window - MIN_STATES_FOR_LYAPUNOV
        if max_evolution_steps < 1:
            continue
        evolution_steps_m = min(evolution_steps, max_evolution_steps)

        # Skip dimension if too few evolution steps
        if evolution_steps_m < 5:
            continue

        num_states = n - embedding_window - evolution_steps_m
        if num_states < MIN_STATES_FOR_LYAPUNOV:
            continue

        # Build embedded state vectors from detrended series
        embedded = np.zeros((num_states, m))
        for i in range(num_states):
            for j in range(m):
                embedded[i, j] = y_detrended[i + j * delay]

        pair_slopes = []

        for i in range(num_states):
            x_t = embedded[i]

            # Find nearest neighbor with temporal separation > tau + 1
            distances = np.linalg.norm(embedded - x_t, axis=1)
            distances[max(0, i - delay - 1) : min(num_states, i + delay + 2)] = np.inf

            if np.all(np.isinf(distances)):
                continue

            neighbor_idx = int(np.nanargmin(distances))
            if not np.isfinite(distances[neighbor_idx]):
                continue

            # Track divergence over time: steps 1 to evolution_steps_m
            log_divergences = []

            for n in range(1, evolution_steps_m + 1):
                i_n = i + n
                j_n = neighbor_idx + n

                if i_n >= num_states or j_n >= num_states:
                    break

                delta_n = np.linalg.norm(embedded[i_n] - embedded[j_n])
                if delta_n < np.finfo(float).eps:
                    break

                log_divergences.append(np.log(delta_n))

            # Require at least 3 divergence points to fit
            if len(log_divergences) < 3:
                continue

            # Fit linear regression: slope is the Lyapunov exponent
            x_fit = np.arange(1, len(log_divergences) + 1)
            y_fit = np.array(log_divergences)

            try:
                slope, _ = np.polyfit(x_fit, y_fit, 1)
                if np.isfinite(slope):
                    pair_slopes.append(slope)
            except:
                continue

        # Compute λ_m: average slope across all valid pairs
        if len(pair_slopes) == 0:
            continue

        # Skip dimension if too few valid pairs
        min_pairs = max(1, MIN_STATES_FOR_LYAPUNOV // 10)
        if len(pair_slopes) < min_pairs:
            continue

        lambda_m = float(np.mean(pair_slopes))
        lambda_dims.append(lambda_m)
        logger.debug(f"Dimension m={m}: λ_m={lambda_m:.4f} (from {len(pair_slopes)} pairs)")

    if len(lambda_dims) == 0:
        return np.nan

    # Return largest exponent from the spectrum
    lambda_max = float(np.max(lambda_dims))
    logger.info(f"Largest Lyapunov exponent: {lambda_max:.4f} (from {len(lambda_dims)} dimensions)")
    
    # Return raw largest Lyapunov exponent
    return lambda_max
