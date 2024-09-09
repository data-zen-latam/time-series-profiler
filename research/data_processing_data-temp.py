#%%
import glob
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL

from ts_profiler.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

#%%
output_path = PROCESSED_DATA_DIR / 'diff_temp_series.parquet.gzip'
# %%
files = glob.glob((RAW_DATA_DIR / '*.csv').__str__())
data_list = [pd.read_csv(file, header=4, index_col=0) for file in files]
data = pd.concat(data_list, axis=1)



#%%

columns = [f.split('/')[-1].split('-')[-1].replace('.csv', '') for f in files]

data.columns = columns

# %%

data = data.reset_index().rename({'Date' : 'date'}, axis = 1)
# %%

data['date'] = pd.to_datetime(data['date'], format='%Y%m').dt.date

data = data[pd.to_datetime(data['date']) >= pd.to_datetime('1910-01-01')]
# %%
mu = 0
std = 1
data['w_noise'] = np.random.normal(mu, std, size = data.shape[0])

# %%
data
# %%
data.to_parquet(output_path, compression='gzip')

# %%
