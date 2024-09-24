#%%

import pandas as pd
import numpy as np

from ts_profiler.config import PROCESSED_DATA_DIR
from statsmodels.tsa.arima.model import ARIMA
from ts_profiler.utils import MeanModel, SeasonalModel, TrendModel, ARIMAModel
#%%

data = pd.read_csv(PROCESSED_DATA_DIR / 'features.csv')
ts = pd.read_csv(PROCESSED_DATA_DIR / 'diff_temp_series.csv')
# %%

### Perfilamiento de caracteristicas y predictibilidad


pd.plotting.scatter_matrix(
    data,
    figsize=(10,10)
    )
# %%
data.describe()
# %%
series = data[data['entropy'].min() == data['entropy']]['series']

ts[series].plot()
# %%

#%%
model_suite = {
    'mean': MeanModel,
    'arima': ARIMAModel,
    'trend': TrendModel,
    'seasonal': SeasonalModel
}

results = pd.DataFrame()


#%%
for serie in ts.set_index('date'):
    y = ts[serie]

    for model_name in model_suite.keys() :

        model = model_suite[model_name](y)
        model.fit()
        yhat = model.predict()

        results = pd.concat([
            results,
            pd.DataFrame([serie, ])
            ])

# %%
def mase(y, yhat):
    return np.mean([
                abs(y[i] - yhat[i]) / (abs(y[i] - y[i - 1]) / len(y) - 1)
                for i in range(1, len(y))
                ])
