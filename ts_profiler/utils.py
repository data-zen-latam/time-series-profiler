from dataclasses import dataclass

import numpy as np
from scipy.stats import entropy, kurtosis, shapiro, skew, yeojohnson
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from tsfresh.feature_extraction.feature_calculators import (
    approximate_entropy,
    fourier_entropy,
)

### Data quality indicators

def range_(x):
    """
    Calculates the range of the series.
    A large range can indicate the presence of outliers or a wide distribution of values.
    """
    return np.max(x) - np.min(x)

def coef_var(x):
    """
    Calculates the coefficient of variation (standard deviation / mean) of the series.
    A high coefficient of variation can indicate a high degree of dispersion or variability in the data.
    """
    return np.std(x, ddof=1) / np.mean(x)


def skewness(x):
    """
    Calculates the skewness of the series.
    Skewness measures the asymmetry of a distribution.
    A non-zero skewness value indicates that the distribution is not symmetric.
    """
    return skew(x[x.notna()])

def n_distinct(x):
    """
    Calculates the number of distinct values in the series.
    A low number of distinct values can indicate a lack of variability or potential data quality issues.
    """
    return len(np.unique(x))

def p_distinct(x):
    """
    Calculates the proportion of distinct values in the series.
    A low proportion of distinct values can indicate a lack of variability or potential data quality issues.
    """
    return len(np.unique(x)) / len(x)

def n_missing(x):
    """
    Calculates the number of missing (NaN) values in the series.
    A high number of missing values can indicate potential data quality issues or incomplete data.
    """
    return len(x[np.isnan(x)])

def p_missing(x):
    """
    Calculates the proportion of missing (NaN) values in the series.
    A high proportion of missing values can indicate potential data quality issues or incomplete data.
    """
    return len(x[np.isnan(x)]) / len(x)

def n_negative(x):
    """
    Calculates the number of negative values in the series.
    A high number of negative values can indicate potential data quality issues or unexpected values.
    """
    return len(x[x < 0])

def p_negative(x):
    """
    Calculates the proportion of negative values in the series.
    A high proportion of negative values can indicate potential data quality issues or unexpected values.
    """
    return len(x[x < 0]) / len(x)

def n_zeros(x):
    """
    Calculates the number of zero values in the series.
    A high number of zero values can indicate potential data quality issues or unexpected values.
    """
    return len(x[x == 0.0])

def p_zeros(x):
    """
    Calculates the proportion of zero values in the series.
    A high proportion of zero values can indicate potential data quality issues or unexpected values.
    """
    return len(x[x == 0.0]) / len(x)

def is_unique(x):
    """
    Checks if all values in the series are unique.
    If all values are unique, it can indicate potential data quality issues or unexpected values.
    """
    return len(np.unique(x)) == len(x)

def quantile_5(x):
    """
    Calculates the 5th percentile of the series.
    Can be used to identify potential outliers or extreme values in the lower tail of the distribution.
    """
    return np.nanquantile(x, 0.05)

def quantile_25(x):
    """
    Calculates the 25th percentile (first quartile) of the series.
    Can be used to identify potential outliers or extreme values in the lower tail of the distribution.
    """
    return np.nanquantile(x, 0.25)

def quantile_50(x):
    """
    Calculates the 50th percentile (median) of the series.
    Can be used to assess the central tendency of the distribution and identify potential outliers or skewness.
    """
    return np.nanquantile(x, 0.5)

def quantile_75(x):
    """
    Calculates the 75th percentile (third quartile) of the series.
    Can be used to identify potential outliers or extreme values in the upper tail of the distribution.
    """
    return np.nanquantile(x, 0.75)

def quantile_95(x):
    """
    Calculates the 95th percentile of the series.
    Can be used to identify potential outliers or extreme values in the upper tail of the distribution.
    """
    return np.nanquantile(x, 0.95)

def iqr(x):
    """
    Calculates the interquartile range (IQR) of the series.
    The IQR is a robust measure of dispersion and can be used to identify potential outliers.
    """
    return np.nanquantile(x, 0.75) - np.nanquantile(x, 0.25)

def mad(x):
    """
    Calculates the mean absolute deviation (MAD) of the series.
    The MAD is a robust measure of dispersion and can be used to identify potential outliers.
    """
    return np.mean(np.abs(x - x.mean()))

def entropy_calculation(x):
    """
    Calculates the entropy of the series.
    Entropy is a measure of uncertainty or randomness in the data.
    A low entropy value can indicate a lack of variability or potential data quality issues.
    """
    v_counts = np.unique(x, return_counts=True)
    return entropy(v_counts[1])

### Time Series forecasteability indicators

def strength_trend(x):
    stl = STL(x, period=13)
    res = stl.fit()

    return (1 - res.resid.var()/(x - res.seasonal).var())

def strength_seasonal(x):
    stl = STL(x, period=13)
    res = stl.fit()
    return  (1 - res.resid.var()/(x - res.trend).var())

def kurtosis_(x):
    """
    Calculates the kurtosis of the series.
    Kurtosis measures the "tailedness" of a distribution.
    A high kurtosis value indicates heavy tails or outliers in the data.
    """
    return kurtosis(x[x.notna()])
### The yeo johnson transformation can be used to normalize, stabilize the variance of a series but tells nothing by itself.
# Kolmogorov Smirnov could be used to measure the difference in the cdfs between original and transformed to detect any difficulties
# stemming from the original data distribution, but you already get this from skewness and kurtosis. NEEDS MORE WORK TO MAKE IT USABLE.

def shapiro_(x):
    """
    Performs the Shapiro-Wilk test for normality on the input time series.
    A low p-value from this test indicates that the time series is not normally distributed,
    which can make it harder to forecast using traditional methods that assume normality.
    """
    return shapiro(x).pvalue

def yeojohnson_(x):
    """
    Applies the Yeo-Johnson transformation to the input time series.
    The Yeo-Johnson transformation is a data normalization technique that can make
    non-normal data more closely resemble a normal distribution.
    A large value of the transformation parameter (returned by this function) suggests
    that the time series is highly non-normal, which can make it harder to forecast.
    """
    return yeojohnson(x[~np.isnan(x)])[1]

def acf1(x):
    """
    Calculates the first-order autocorrelation coefficient of the input time series.
    A high autocorrelation value indicates that the time series is highly correlated
    with its own past values, which can make it easier to forecast using time series
    models that capture this autocorrelation structure.
    """
    return np.corrcoef(x[0:-1], x[1:])[0, 1]

def approximate_entropy_(x):
    """
    Calculates the approximate entropy of the input time series.
    Approximate entropy is a measure of the complexity and irregularity of a time series.
    A high approximate entropy value suggests that the time series is highly irregular
    and complex, which can make it harder to forecast accurately.
    """
    return approximate_entropy(x, 2, 0.2 * x.std())

def fourier_entropy_(x):
    """
    Calculates the Fourier entropy of the input time series.
    Fourier entropy is a measure of the spectral complexity of a time series.
    A high Fourier entropy value indicates that the time series has a complex
    frequency spectrum, which can make it harder to forecast using traditional
    time series models that assume simpler spectral structures.
    """
    return fourier_entropy(x, bins=10)

def adf(x):
    """
    Performs the Augmented Dickey-Fuller (ADF) test for stationarity on the input time series.
    A low p-value from this test indicates that the time series is stationary,
    which can make it easier to forecast using traditional time series models
    that assume stationarity.
    """
    return adfuller(x)[1]

### Mutual information missing.

@dataclass
class MeanModel:
    x: np.ndarray

    def fit(self) -> None:
        self.mean = np.array([self.x.mean()] * self.x.shape[0])

    def predict(self) -> np.ndarray:
        return self.mean

@dataclass
class TrendModel:
    x: np.ndarray

    def fit(self) -> None:
        stl = STL(self.x, period=13)
        res = stl.fit()
        self.trend = res.trend

    def predict(self) -> np.ndarray:
        return self.trend


@dataclass
class SeasonalModel:
    x: np.ndarray

    def fit(self) -> None:
        stl = STL(self.x, period=13)
        res = stl.fit()
        self.seasonal = res.seasonal

    def predict(self) -> np.ndarray:
        return self.seasonal



@dataclass
class ARIMAModel:
    x: np.ndarray
    model = ARIMA

    def fit(self) -> None:
        self.model = self.model(self.x, order = (1,1,1)).fit()

    def predict(self) -> np.ndarray:
        return np.array([])
        # return self.model.predict()

