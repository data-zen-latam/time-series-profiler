"""Utility functions for parsing and preprocessing single time series data."""

import pandas as pd


def load_csv(file_path):
    """
    Load a CSV file into a DataFrame.
    
    Args:
        file_path: str, path to CSV file
        
    Returns:
        pd.DataFrame: loaded data
    """
    return pd.read_csv(file_path)


def parse_timestamp(df):
    """
    Parse and validate timestamp column.
    
    Args:
        df: pd.DataFrame with 'timestamp' column
        
    Returns:
        pd.DataFrame: DataFrame with parsed timestamp
    """
    raise NotImplementedError("To be implemented.")


def sliding_window(values, window_size):
    """
    Create sliding windows from a time series.
    
    Args:
        values: array-like, time series values
        window_size: int, window size
        
    Yields:
        array-like: windows of the specified size
    """
    raise NotImplementedError("To be implemented.")
