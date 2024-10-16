from pathlib import Path
import numpy as np

from ts_profiler.config import PROCESSED_DATA_DIR
from ts_profiler.utils import *

funcs = {
    'sum': np.sum,
    'mean': np.mean,
    'std': np.std,
    'variance': np.var,
    'min': np.min,
    'max': np.max,
    'kurtosis': kurtosis,
    'skewness': skew,
    'mad': mad,
    '5%': quantile_5,
    '25%': quantile_25,
    '50%': quantile_50,
    '75%': quantile_75,
    '95%': quantile_95,
    'iqr': iqr,
    'range': range_,
    'n': len,
    'n_distinct': n_distinct,
    'p_distinct': p_distinct,
    'is_unique': is_unique,
    'n_missing': n_missing,
    'p_missing': p_missing,
    'n_negative': n_negative,
    'p_negative': p_negative,
    'n_zeros': n_zeros,
    'p_zeros': p_zeros,
    'approx_entropy': approximate_entropy_,
    'adfuller': adf,
    # 'mutual_info',
    'cv': coef_var,
    'entropy': entropy_calculation,
    'strength_trend': strength_trend,
    'strength_seasonality': strength_seasonal,
    'autocorr_lag1': acf1,
    'yeojohnson': yeojohnson_
}