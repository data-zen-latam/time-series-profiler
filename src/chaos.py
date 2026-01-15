"""Chaotic behavior metrics module.

Compute chaotic behavior metrics such as the largest Lyapunov exponent via delay embedding.
"""

import logging

import numpy as np
from statsmodels.tsa.stattools import acf

from .utils import detrend_series

logger = logging.getLogger(__name__)

MIN_STATES_FOR_LYAPUNOV = 10


def _estimate_delay_autocorrelation(y, threshold=0.3):
    """Estimate optimal delay using autocorrelation method.
    
    Detrends the series first, then finds the first lag where autocorrelation 
    drops below threshold or reaches its first minimum.
    
    Args:
        y: array, time series values
        threshold: float, ACF threshold for delay selection
        
    Returns:
        int: estimated delay τ
    """
    # Detrend to remove trend and seasonality before computing ACF
    y_detrended = detrend_series(y)
    
    # Guard against constant/zero-variance series
    if np.std(y_detrended) == 0:
        return 1
    
    n = len(y_detrended)
    max_lag = min(100, n // 4)  # Heuristic. Don't search beyond 1/4 of series length.
    
    if max_lag < 2:
        return 1
    
    # Compute ACF using statsmodels (efficient implementation)
    # Returns ndarray of shape (nlags+1,) when alpha=None and qstat=False
    acf_vals = acf(y_detrended, nlags=max_lag)
    
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


def estimate_embedding_dimension_cao(y, max_dim=10, tau=None, e1_threshold=1.1):
    """Estimate optimal embedding dimension using Cao's method with auto-detected elbow.
    
    Finds minimum m where E1(m) plateaus (ratio < threshold), indicating sufficient
    embedding has been achieved.
    
    Args:
        y: array-like, time series values
        max_dim: int, maximum dimension to test (default: 10)
        tau: int or None, time delay (default: None, auto-estimated via ACF)
        e1_threshold: float, threshold for E1 ratio to detect plateau (default: 1.1)
        
    Returns:
        int: estimated optimal embedding dimension (minimum 2)
    """
    y = np.asarray(y, dtype=float)
    
    if tau is None:
        tau = max(1, _estimate_delay_autocorrelation(y))
    
    E1, _ = cao_e1_e2(y, max_dim=max_dim, tau=tau)
    
    if E1.size == 0 or np.all(np.isnan(E1)):
        return 2

    # E1[i] corresponds to ratio E(m=i+2)/E(m=i+1); plateau at i -> embedding dimension i+2
    optimal_dim = 2
    for i in range(E1.size - 1):
        if np.isfinite(E1[i]) and E1[i] < e1_threshold:
            if np.isfinite(E1[i + 1]) and E1[i + 1] < e1_threshold:
                optimal_dim = i + 2
                break
    
    if optimal_dim == 2:
        valid_e1 = E1[np.isfinite(E1) & (E1 > 0)]
        if valid_e1.size > 0:
            min_idx = int(np.nanargmin(E1))
            optimal_dim = min_idx + 2 # m = i + 2
    
    logger.info(f"Cao's method estimated optimal embedding dimension: {optimal_dim} (tau={tau})")
    return optimal_dim


def largest_lyapunov_exponent(values, embedding_dim=None, delay=None, evolution_steps=None):
    """Compute the largest Lyapunov exponent via delay embedding.
    
    Reconstructs the phase space using time-delay embedding with automatic delay
    estimation via ACF on detrended data, then tracks the divergence of initially
    close trajectories. Returns a normalized chaos score in [0, 1].
    
    Args:
        values: array-like, time series values
        embedding_dim: int or None, embedding dimension m (default: None, auto-estimated via Cao's method)
        delay: int or None, time delay τ for embedding (default: None, auto-estimated via ACF on detrended series)
        evolution_steps: int or None, number of steps Δt to track divergence (default: None, 5% of series length, minimum 10)
        (neighbors use a temporal exclusion window of τ+1 to avoid trivial autocorrelated matches)
        
    Returns:
        float: chaos score in [0, 1] via sigmoid transformation of largest Lyapunov exponent
               < 0.5 indicates stable behavior (negative λ)
               > 0.5 indicates chaotic behavior (positive λ)
               = 0.5 indicates neutral (λ = 0)
               Returns NaN if insufficient data or non-finite values.
    """
    y = np.asarray(values, dtype=float)
    n = y.size
    
    if not np.isfinite(y).all():
        return np.nan
    
    # Guard against constant/zero-variance series
    if np.std(y) == 0:
        return 0.0  # Constant series is maximally stable (λ ≤ 0)
    
    # Auto-estimate evolution_steps as percentage of series length if not specified
    if evolution_steps is None:
        evolution_steps = max(10, int(n * 0.05))  # 5% of series length, minimum 10
    
    # Auto-estimate embedding dimension using Cao's method if not specified
    if embedding_dim is None:
        embedding_dim = estimate_embedding_dimension_cao(y, max_dim=10, tau=delay)
    
    # Validate input
    if n < 100 * embedding_dim:
        return np.nan  # Insufficient data for reliable estimation
    
    # Auto-estimate delay if not provided
    if delay is None:
        delay = _estimate_delay_autocorrelation(y)

    # Use delay (tau) to avoid trivial neighbors close in time; add 1 to be slightly stricter than exact tau, i.e., approximately linear independent delays.
    effective_min_separation = delay + 1
    
    # Time-delay embedding: x_t = (y_t, y_{t+τ}, ..., y_{t+(m-1)τ})
    embedding_window = (embedding_dim - 1) * delay

    # Clamp evolution_steps so enough states remain
    max_evolution_steps = n - embedding_window - MIN_STATES_FOR_LYAPUNOV
    if max_evolution_steps < 1:
        return np.nan
    evolution_steps = min(evolution_steps, max_evolution_steps)

    num_states = n - embedding_window - evolution_steps
    
    if num_states < MIN_STATES_FOR_LYAPUNOV:
        return np.nan  # Not enough states to estimate
    
    # Build embedded state vectors
    embedded = np.zeros((num_states, embedding_dim))
    for i in range(num_states):
        for j in range(embedding_dim):
            embedded[i, j] = y[i + j * delay]
    
    # Track divergence for each state
    divergences = []
    
    for i in range(num_states):
        x_t = embedded[i]
        
        # Find nearest neighbor with sufficient temporal separation
        distances = np.linalg.norm(embedded - x_t, axis=1)
        
        # Exclude self and temporally close neighbors
        valid_neighbors = np.where(
            (distances > 0) & 
            (np.abs(np.arange(num_states) - i) > effective_min_separation)
        )[0]
        
        if len(valid_neighbors) == 0:
            continue
        
        # Get nearest valid neighbor
        neighbor_idx = valid_neighbors[np.argmin(distances[valid_neighbors])]
        
        # Initial separation
        delta_0 = distances[neighbor_idx]
        
        if delta_0 < np.finfo(float).eps:
            continue  # Degenerate pair
        
        # Track evolution over Δt steps
        i_evolved = i + evolution_steps
        neighbor_evolved = neighbor_idx + evolution_steps
        
        # Check if evolved states are within bounds
        if i_evolved >= num_states or neighbor_evolved >= num_states:
            continue
        
        x_t_evolved = embedded[i_evolved]
        x_neighbor_evolved = embedded[neighbor_evolved]
        
        # Final separation after Δt steps
        delta_t = np.linalg.norm(x_t_evolved - x_neighbor_evolved)
        
        if delta_t < np.finfo(float).eps:
            continue  # No divergence to measure
        
        # Compute local Lyapunov exponent: λ = (1/Δt) * log(δ(Δt) / δ_0)
        local_lambda = (1.0 / evolution_steps) * np.log(delta_t / delta_0)
        divergences.append(local_lambda)
    
    if len(divergences) == 0:
        return np.nan
    
    # Take maximum over all tracked pairs (largest Lyapunov exponent)
    lambda_max = np.max(divergences)
    
    # Return raw largest Lyapunov exponent
    return float(lambda_max)
