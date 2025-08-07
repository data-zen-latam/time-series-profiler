#%%
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ts_profiler.config import (
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    descriptive_stats,
    features,
    value_summary,
)
from ts_profiler.features import funcs

#%%
# input_path = PROCESSED_DATA_DIR / 'diff_temp_series.parquet.gzip'
PROCESSED_DATA_DIR = Path(os.getcwd()).parents[0] / 'data/processed/'
REPORTS_DIR = Path(os.getcwd()) / 'reports/'
input_path = PROCESSED_DATA_DIR / 'diff_temp_series.parquet.gzip'
output_path = REPORTS_DIR / 'timeseries_diagnostics.html'
# %%
data = pd.read_parquet(input_path)

#%%
data.info()
#%%

data.head()

#%%
result = pd.DataFrame()
for c in data.drop(['date'], axis=1):
    stats = {
        k: f(data[c])
        for k,f in funcs.items()
    }

    stats['series'] = c

    result = pd.concat([
        result,
        pd.DataFrame([stats], index= [0])
    ])

#%%
aux = data.melt(id_vars='date', var_name='series')

result['upper_threshold'] = result['75%'] + 1.5 * result['iqr']
result['lower_threshold'] = result['25%'] - 1.5 * result['iqr']

aux = aux.merge(result[['series', 'upper_threshold', 'lower_threshold']], how='left', on='series')

aux['p_outliers'] = (aux['value'] > aux['upper_threshold']) | (aux['value'] < aux['lower_threshold'])
aux = aux.groupby(['series'])['p_outliers'].mean()

result = pd.concat([result.set_index('series'), aux], axis=1)


#%%
score_cols = [   # 'cv',
    'p_distinct',
    'p_missing',
    'p_zeros',
    'p_outliers',
    'entropy',
]

aux = result[score_cols]

aux['score_distinct'] = aux['p_distinct']
aux['score_missing'] = 1 - aux['p_missing']
aux['score_zeros'] = 1 - aux['p_zeros']
aux['score_outliers'] = 1 - aux['p_outliers']
aux['score_entropy'] = 1 - (aux['entropy'] / aux.loc['w_noise', 'entropy'])
# TODO : Corregir distribucion de white noise con parametros para cada series / monte carlo de series con entropia
#%%
aux['completeness_score'] = aux.loc[:, aux.columns.str.contains('score')].mean(axis=1)

scores_ = aux.copy(deep=True)
#%%
### Forecastability
score_forecast = [ 'entropy',
 'strength_trend',
 'strength_seasonality',
 'kurtosis',
 'adfuller',
 'shapiro',
 'fourier_entropy'
 ]

aux = result[score_forecast]
aux['flag_adf'] = aux['adfuller'] > 0.05 # Not Reject Null Hypothesis ie There is Unit Root
aux['flag_shapiro'] = aux['shapiro'] < 0.05 # Reject Null Hypothesis ie Non Normal Distribution
aux['flag_kurtosis'] = aux['kurtosis'] > 3

aux['score_flags'] = 1 - aux.loc[:, aux.columns.str.contains('flag')].sum(axis=1) / np.sum(aux.columns.str.contains('flag'))
aux['score_trend'] = aux['strength_trend']
aux['score_seasonality'] = aux['strength_seasonality']
aux['score_entropy'] = 1 - (aux['entropy'] / aux.loc['w_noise', 'entropy'])


aux['predictability_score'] = aux.loc[:, aux.columns.str.contains('score')].mean(axis=1)

scores_ = pd.concat([scores_, aux], axis=1)

#%%
import seaborn as sns

sns.scatterplot(scores_, x='completeness_score', y='predictability_score')

#%%
result[features].to_csv(PROCESSED_DATA_DIR / 'features.csv', index=False)
result[['series'] + descriptive_stats].to_csv(PROCESSED_DATA_DIR / 'descriptive_stats.csv', index=False)
result[['series'] + value_summary].to_csv(PROCESSED_DATA_DIR / 'values_summary.csv', index=False)

### Pending HTML.Ydataprofiling bug.

#%%
from ydata_profiling import ProfileReport

# %%
data['date'] = data['date'].astype('datetime64[ns]')


profile = ProfileReport(data, tsmode=True, sortby='date')
# %%
profile.to_file(output_path.__str__())
# %%
