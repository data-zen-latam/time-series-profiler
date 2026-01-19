"""Unit tests for data quality metrics.

Demonstrates pytest features:
- @pytest.mark.parametrize for testing edge cases
- pytest.approx for float comparisons
- fixtures for shared test data
"""

import numpy as np
import pytest

from src.data_quality import (aggregate_quality_score, missing_value_ratio,
                              outlier_count_ratio, unique_value_ratio)

# Fixtures are defined in conftest.py and automatically available


class TestMissingValueRatio:
    """Test missing value ratio computation."""
    
    def test_no_missing_values(self, perfect_series):
        """Series with no missing values should return 0.0."""
        ratio = missing_value_ratio(perfect_series)
        assert ratio == pytest.approx(0.0)
        
    def test_all_missing_values(self):
        """Series with all NaN should return 1.0."""
        series = np.array([np.nan, np.nan, np.nan, np.nan])
        ratio = missing_value_ratio(series)
        assert ratio == pytest.approx(1.0)
    
    # Parametrized test: Different missing ratios
    @pytest.mark.parametrize("n_total,n_missing,expected", [
        (10, 0, 0.0),
        (10, 5, 0.5),
        (10, 10, 1.0),
        (100, 25, 0.25),
    ])
    def test_partial_missing_values(self, n_total, n_missing, expected):
        """Series with various missing ratios should return correct values."""
        series = np.ones(n_total)
        series[:n_missing] = np.nan
        ratio = missing_value_ratio(series)
        assert ratio == pytest.approx(expected)
    
    # Parametrized test: Different types of invalid values
    @pytest.mark.parametrize("invalid_value", [np.inf, -np.inf])
    def test_inf_values_counted_as_missing(self, invalid_value):
        """Inf values should be counted as missing."""
        series = np.array([1.0, invalid_value, 3.0, invalid_value, 5.0])
        ratio = missing_value_ratio(series)
        assert ratio == pytest.approx(0.4)  # 2 out of 5
        
    def test_mixed_nan_and_inf(self):
        """Mix of NaN and inf should all count as missing."""
        series = np.array([1.0, np.nan, np.inf, -np.inf, 5.0])
        ratio = missing_value_ratio(series)
        assert ratio == pytest.approx(0.6)  # 3 out of 5
        
    def test_returns_float(self, perfect_series):
        """Missing value ratio should return float."""
        ratio = missing_value_ratio(perfect_series)
        assert isinstance(ratio, (float, np.floating))


class TestOutlierCountRatio:
    """Test outlier detection."""
    
    def test_no_outliers(self, normal_distribution):
        """Normal distribution should have few/no outliers."""
        ratio = outlier_count_ratio(normal_distribution, threshold=3.0)
        assert ratio < 0.01  # Less than 1% for 3-sigma
        
    def test_with_clear_outliers(self, series_with_outliers):
        """Series with obvious outliers should detect them."""
        # With very small series, z-score may not detect due to high variance
        ratio = outlier_count_ratio(series_with_outliers, threshold=2.0)
        assert ratio > 0.1  # Should detect at least some outliers
        
    def test_constant_series_no_outliers(self, constant_series):
        """Constant series should have no outliers (guard for zero std)."""
        ratio = outlier_count_ratio(constant_series)
        assert ratio == pytest.approx(0.0)
    
    # Parametrized test: Different thresholds
    @pytest.mark.parametrize("threshold,expected_range", [
        (2.0, (0.1, 0.5)),  # Stricter: more outliers detected
        (4.0, (0.0, 0.3)),  # Lenient: fewer outliers
    ])
    def test_threshold_parameter(self, series_with_outliers, threshold, expected_range):
        """Different thresholds should detect different outlier counts."""
        ratio = outlier_count_ratio(series_with_outliers, threshold=threshold)
        min_expected, max_expected = expected_range
        assert min_expected <= ratio <= max_expected
        
    def test_returns_float(self, series_with_outliers):
        """Outlier ratio should return float."""
        ratio = outlier_count_ratio(series_with_outliers)
        assert isinstance(ratio, (float, np.floating))
        
    def test_series_with_nans_filtered(self):
        """NaN values should be filtered before outlier detection."""
        series = np.array([1, 2, np.nan, 4, 5, 100])
        ratio = outlier_count_ratio(series)
        assert not np.isnan(ratio)


class TestUniqueValueRatio:
    """Test unique value ratio computation."""
    
    def test_all_unique_values(self, perfect_series):
        """All unique values should return 1.0."""
        ratio = unique_value_ratio(perfect_series)
        assert ratio == pytest.approx(1.0)
        
    def test_all_same_values(self, constant_series):
        """All same values should return very low ratio."""
        ratio = unique_value_ratio(constant_series)
        assert ratio == pytest.approx(0.002)  # 1/500 (constant_series has 500 values)
    
    # Parametrized test: Different duplication patterns
    @pytest.mark.parametrize("series,expected", [
        (np.array([1, 1, 2, 2, 3, 3]), 0.5),  # 3 unique / 6 total
        (np.array([1, 2, 3, 4, 5]), 1.0),      # all unique
        (np.array([5, 5, 5, 5]), 0.25),        # 1 unique / 4 total
    ])
    def test_various_duplication_levels(self, series, expected):
        """Different duplication patterns should return correct ratios."""
        ratio = unique_value_ratio(series)
        assert ratio == pytest.approx(expected)
        
    def test_nans_filtered(self):
        """NaN values should be filtered out."""
        series = np.array([1, 2, np.nan, 3, np.nan, 4])
        ratio = unique_value_ratio(series)
        # 4 unique values (1,2,3,4) out of 4 valid values
        assert ratio == pytest.approx(1.0)
        
    def test_returns_float(self, perfect_series):
        """Unique value ratio should return float."""
        ratio = unique_value_ratio(perfect_series)
        assert isinstance(ratio, (float, np.floating))


class TestAggregateQualityScore:
    """Test quality score aggregation."""
    
    def test_perfect_quality(self, perfect_series):
        """Perfect data should have quality score close to 1.0."""
        score = aggregate_quality_score(perfect_series)
        assert score == pytest.approx(1.0)
        
    def test_constant_series_low_quality(self, constant_series):
        """Constant series should have low quality (low uniqueness)."""
        score = aggregate_quality_score(constant_series)
        assert score < 0.1  # Very low due to unique_ratio = 0.01
    
    # Parametrized test: Various quality degradations
    @pytest.mark.parametrize("series_type,max_score", [
        ("with_missing", 0.8),
        ("constant", 0.1),
    ])
    def test_degraded_quality_scenarios(self, series_type, max_score):
        """Various data quality issues should reduce score appropriately."""
        if series_type == "with_missing":
            series = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        elif series_type == "constant":
            series = np.ones(100)
        else:
            raise ValueError(f"Unknown series_type: {series_type}")
        
        score = aggregate_quality_score(series)
        assert score < max_score
        
    def test_score_bounded_zero_to_one(self, normal_distribution):
        """Quality score should be between 0 and 1."""
        score = aggregate_quality_score(normal_distribution)
        assert 0.0 <= score <= 1.0
        
    def test_returns_float(self, perfect_series):
        """Quality score should return float."""
        score = aggregate_quality_score(perfect_series)
        assert isinstance(score, (float, np.floating))
        
    def test_all_missing_returns_zero(self):
        """All missing values should result in zero quality."""
        series = np.array([np.nan, np.nan, np.nan])
        score = aggregate_quality_score(series)
        assert score == pytest.approx(0.0)
        
    def test_better_quality_higher_score(self):
        """Better quality data should have higher score."""
        good_series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        bad_series = np.array([1, 1, 1, np.nan, np.nan, 100, 200, 1, 1, 1])
        
        score_good = aggregate_quality_score(good_series)
        score_bad = aggregate_quality_score(bad_series)
        
        assert score_good > score_bad
