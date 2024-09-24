import numpy as np
from scipy.stats import entropy, yeojohnson
from scipy.stats import kurtosis, skew
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from tsfresh.feature_extraction.feature_calculators import approximate_entropy
from statsmodels.tsa.arima.model import ARIMA
from dataclasses import dataclass

### Add fourier entropy.

def range_(x):
    return np.max(x) - np.min(x)

def coef_var(x):
    return np.std(x, ddof=1) / np.mean(x)

def kurtosis_(x):
    return kurtosis(x[x.notna()])

def skewness(x):
    return skew(x[x.notna()])

def n_distinct(x):
    return len(np.unique(x))

def p_distinct(x):
    return len(np.unique(x)) / len(x)

def n_missing(x):
    return len(x[np.isnan(x)])

def p_missing(x):
    return len(x[np.isnan(x)]) / len(x)

def n_negative(x):
    return len(x[x < 0])

def p_negative(x):
    return len(x[x < 0]) / len(x)

def n_zeros(x):
    return len(x[x == 0.0])

def p_zeros(x):
    return len(x[x == 0.0]) / len(x)

def is_unique(x):
    return len(np.unique(x)) == len(x)

def quantile_5(x):
    return np.nanquantile(x, 0.05)

def quantile_25(x):
    return np.nanquantile(x, 0.25)

def quantile_50(x):
    return np.nanquantile(x, 0.5)

def quantile_75(x):
    return np.nanquantile(x, 0.75)

def quantile_95(x):
    return np.nanquantile(x, 0.95)

def iqr(x):
    return np.nanquantile(x, 0.75) - np.nanquantile(x, 0.25)

def mad(x):
    return (x - x.mean()).abs().mean()

def entropy_calculation(x):
    v_counts = np.unique(x, return_counts=True)
    return entropy(v_counts[1])

def strength_trend(x):
    stl = STL(x, period=13)
    res = stl.fit()

    return (1 - res.resid.var()/(x - res.seasonal).var())

def strength_seasonal(x):
    stl = STL(x, period=13)
    res = stl.fit()
    return  (1 - res.resid.var()/(x - res.trend).var())

def yeojohnson_(x):
    return yeojohnson(x[x.notna()])[1]

def acf1(x):
    return np.corrcoef(x[0:-1], x[1:])[0,1]

def approximate_entropy_(x):
    return approximate_entropy(x, 2, 0.2*x.std())

def adf(x):
    return adfuller(x)[1]

@dataclass
class MeanModel:
    x: np.array

    def fit(self) -> None:
        self.mean = np.array([self.x.mean()] * self.x.shape[0])

    def predict(self) -> np.array:
        return self.mean

@dataclass
class TrendModel:
    x: np.array

    def fit(self) -> None:
        stl = STL(self.x, period=13)
        res = stl.fit()
        self.trend = res.trend

    def predict(self) -> np.array:
        return self.trend


@dataclass
class SeasonalModel:
    x: np.array

    def fit(self) -> None:
        stl = STL(self.x, period=13)
        res = stl.fit()
        self.seasonal = res.seasonal

    def predict(self) -> np.array:
        return self.seasonal



@dataclass
class ARIMAModel:
    x: np.array
    model = ARIMA

    def fit(self) -> None:
        self.model = self.model(self.x, order = (1,1,1)).fit()

    def predict(self) -> np.array:
        return self.model.predict()

