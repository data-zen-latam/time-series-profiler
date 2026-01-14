"""Chaos metrics module.

Compute chaos metrics such as the largest Lyapunov exponent via delay embedding.
"""


def largest_lyapunov_exponent(values, embedding_dim=5, delay=1):
    """
    Compute the largest Lyapunov exponent via delay embedding.
    
    Args:
        values: array-like, time series values
        embedding_dim: int, embedding dimension for phase space reconstruction
        delay: int, delay for embedding
        
    Returns:
        float: estimated largest Lyapunov exponent
        
    Raises:
        ValueError: if series is too short or sparse
    """
    raise NotImplementedError("To be implemented.")
