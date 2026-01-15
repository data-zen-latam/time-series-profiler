"""Data quality profiling module.

Extract and aggregate data quality metrics from time series.
"""

import numpy as np
from scipy import stats


def missing_value_ratio(values):
    """
    Compute the ratio of missing values.
    
    Args:
        values: array-like, time series values
        
    Returns:
        float: ratio of missing values [0, 1]
    """
    values = np.asarray(values)
    n_missing = np.sum(~np.isfinite(values))
    return float(n_missing / len(values))


def outlier_count_ratio(values, threshold=3.0):
    """
    Compute the ratio of outliers detected (e.g., >threshold std devs from mean).
    
    Args:
        values: array-like, time series values
        threshold: float, standard deviation threshold
        
    Returns:
        float: ratio of outliers [0, 1]
    """
    values = np.asarray(values, dtype=float)
    valid = values[np.isfinite(values)]
    
    if len(valid) == 0:
        return 0.0
    
    z_scores = np.abs((valid - np.mean(valid)) / np.std(valid))
    n_outliers = np.sum(z_scores > threshold)
    return float(n_outliers / len(valid))



def unique_value_ratio(values):
    """
    Compute the ratio of unique values to total values.
    
    Args:
        values: array-like, time series values
        
    Returns:
        float: ratio of unique values [0, 1]
    """
    values = np.asarray(values)
    valid = values[np.isfinite(values)]
    
    if len(valid) == 0:
        return 0.0
    
    n_unique = len(np.unique(valid))
    return float(n_unique / len(valid))


def aggregate_quality_score(values):
    """
    Aggregate individual quality metrics into a single score [0, 1].
    
    Higher score indicates better data quality:
    - 0 missing values (bonus)
    - 0 outliers (bonus)
    - High unique value ratio (no repetition)
    
    Args:
        values: array-like, time series values
        
    Returns:
        float: aggregated data quality score [0, 1]
    """
    # Individual quality components
    missing_ratio = missing_value_ratio(values)
    outlier_ratio = outlier_count_ratio(values)
    unique_ratio = unique_value_ratio(values)
    
    # Weighted combination: (1 - problems) * uniqueness
    # Missing and outliers degrade quality, uniqueness is a feature
    quality = (1 - missing_ratio) * (1 - outlier_ratio) * unique_ratio
    
    # Clamp to [0, 1]
    return float(np.clip(quality, 0.0, 1.0))
