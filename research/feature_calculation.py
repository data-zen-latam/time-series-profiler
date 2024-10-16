#%%
import pandas as pd

import numpy as np
from ts_profiler.config import  PROCESSED_DATA_DIR, REPORTS_DIR, features, descriptive_stats, value_summary
from ts_profiler.features import funcs

import os
from pathlib import Path

#%%
# input_path = PROCESSED_DATA_DIR / 'diff_temp_series.parquet.gzip'
PROCESSED_DATA_DIR = Path(os.getcwd()).parents[0] / 'data/processed/'
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
