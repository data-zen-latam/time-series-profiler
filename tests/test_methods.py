#   ---------------------------------------------------------------------------------
#   Copyright (c) Microsoft Corporation. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   ---------------------------------------------------------------------------------
"""Test time series profiler functionality."""

import numpy as np
import pandas as pd
import pytest

from ts_profiler import config, features, utils


def test_config_imports():
    """Test that config module imports correctly."""
    assert hasattr(config, 'PROJ_ROOT')
    assert hasattr(config, 'features')
    assert hasattr(config, 'descriptive_stats')


def test_basic_time_series_creation():
    """Test basic time series data creation."""
    # Create a simple time series
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    values = np.random.randn(100)
    ts = pd.Series(values, index=dates)
    
    assert len(ts) == 100
    assert isinstance(ts.index, pd.DatetimeIndex)


def test_features_module_exists():
    """Test that features module can be imported."""
    assert features is not None


def test_utils_module_exists():
    """Test that utils module can be imported."""
    assert utils is not None


def test_tsfresh_integration():
    """Test that tsfresh can be imported and used."""
    try:
        import tsfresh
        from tsfresh import extract_features
        from tsfresh.utilities.dataframe_functions import impute
        
        # Create sample time series data in tsfresh format
        df = pd.DataFrame({
            'id': [1] * 50 + [2] * 50,
            'time': list(range(50)) + list(range(50)),
            'value': np.random.randn(100)
        })
        
        # Extract features (this might take a moment)
        features_df = extract_features(df, column_id='id', column_sort='time')
        
        # Impute NaN values
        features_df = impute(features_df)
        
        assert len(features_df) == 2  # Should have features for 2 time series
        assert len(features_df.columns) > 0  # Should have extracted some features
        
    except ImportError:
        pytest.skip("tsfresh not available")


@pytest.mark.parametrize("data_size", [10, 100, 1000])
def test_time_series_different_sizes(data_size):
    """Test with different time series sizes."""
    ts = pd.Series(np.random.randn(data_size))
    assert len(ts) == data_size
