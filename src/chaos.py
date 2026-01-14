"""Chaos metrics module.

Compute chaos metrics such as the largest Lyapunov exponent via delay embedding.
"""

import logging
import numpy as np
from statsmodels.tsa.stattools import acf
from utils import detrend_series

logger = logging.getLogger(__name__)


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


def largest_lyapunov_exponent(values, embedding_dim=5, delay=None, evolution_steps=10, min_temporal_separation=3):
    """Compute the largest Lyapunov exponent via delay embedding.
    
    Reconstructs the phase space using time-delay embedding with automatic delay
    estimation via ACF on detrended data, then tracks the divergence of initially
    close trajectories. Returns a normalized chaos score in [0, 1].
    
    Args:
        values: array-like, time series values
        embedding_dim: int, embedding dimension m (default: 5)
        delay: int or None, time delay τ for embedding (default: None, auto-estimated via ACF on detrended series)
        evolution_steps: int, number of steps Δt to track divergence in series starting from similar points (default: 10)
        min_temporal_separation: int, minimum time separation between state pairs (default: 3, heuristic.)
        
    Returns:
        float: chaos score in [0, 1] via sigmoid transformation of largest Lyapunov exponent
               < 0.5 indicates stable behavior (negative λ)
               > 0.5 indicates chaotic behavior (positive λ)
               = 0.5 indicates neutral (λ = 0)
               Returns NaN if insufficient data (<100×m points) or non-finite values.
    """
    y = np.asarray(values, dtype=float)
    n = y.size
    
    # Validate input
    if n < 100 * embedding_dim:
        return np.nan  # Insufficient data for reliable estimation
    
    if not np.isfinite(y).all():
        return np.nan
    
    # Auto-estimate delay if not provided
    if delay is None:
        delay = max(min_temporal_separation, _estimate_delay_autocorrelation(y))
    
    # Time-delay embedding: x_t = (y_t, y_{t+τ}, ..., y_{t+(m-1)τ})
    embedding_window = (embedding_dim - 1) * delay
    num_states = n - embedding_window - evolution_steps
    
    if num_states < 10:
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
            (np.abs(np.arange(num_states) - i) > min_temporal_separation)
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
    
    # Transform to [0, 1] via sigmoid: 0.5 threshold separates chaotic (>0.5) from stable (<0.5)
    chaos_score = 1.0 / (1.0 + np.exp(-lambda_max))
    
    return float(chaos_score)
