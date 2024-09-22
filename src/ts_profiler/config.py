from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


PROJ_ROOT = Path(__file__).resolve().parents[2]

REPORTS_DIR = PROJ_ROOT / "reports"
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"


features = [
    'series',
    'approx_entropy',
    # 'mutual_info',
    'entropy',
    'strength_trend',
    'strength_seasonality',
    'autocorr_lag1',
    'yeojohnson'
]

value_summary= [
    'n_distinct',
    'p_distinct',
    'is_unique',
    'n_missing',
    'n',
    'p_missing',
    'n_negative',
    'p_negative',
    'n_zeros',
    'p_zeros',
]

descriptive_stats = ['mean', 'std',
'variance', 'min', 'max', 'kurtosis',
'cv', 'skewness', 'sum', 'mad', 'adfuller',
# 'chi_squared',
'range', '5%', '25%', '50%', '75%', '95%', 'iqr',
]