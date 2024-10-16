#%%
import pandas as pd
from ts_profiler.config import RAW_DATA_DIR
from ts_profiler.config import features, descriptive_stats, value_summary
from ts_profiler.features import funcs
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
#%%
input_path = RAW_DATA_DIR / 'M3C.xls'
data = {}


data['year'] = pd.read_excel(input_path, sheet_name='M3Year')
data['quart'] = pd.read_excel(input_path, sheet_name='M3Quart')
data['month'] = pd.read_excel(input_path, sheet_name='M3Month')
data['other'] = pd.read_excel(input_path, sheet_name='M3Other')
# %%


def extract_m3_series(data, period):
    meta_data = data[period].iloc[:, :6]
    series = []
    series = series + [
        (serie, data[period].iloc[ i, 6:].to_numpy().astype(float))
        for i, serie in enumerate(meta_data['Series'])#: print(i, serie)
    ]

    return series, meta_data



extraction = [(extract_m3_series(data, period)) for period in ['year', 'month', 'quart', 'other']]


aux = [serie for serie, metadata in extraction]
series = list(itertools.chain(*aux))


series = {name : serie for name, serie in series}
metadata = pd.concat([ metadata for serie, metadata in extraction])


#%%
result = pd.DataFrame()
for k, serie in series.items():
    stats = {
        k: f(serie[~np.isnan(serie)])
        for k,f in funcs.items()
    }

    stats['series'] = k

    result = pd.concat([
        result,
        pd.DataFrame([stats], index= [0])
    ])


# %%
data_features = result[features]
# %%

data_features


pd.plotting.scatter_matrix(
    data_features,
    figsize=(10,10)
    )

x = data_features.iloc[:, 2:]

pca = PCA()



pca.fit(x)

pca.explained_variance_ratio_

x_pca = pca.transform(x)
#%%
coef = pca.components_
sns.scatterplot(x=x_pca[:,0], y=x_pca[:,1])

# %%
plt.scatter(x=x_pca[:,0], y=x_pca[:,1])
plt.arrow(0,0, coef[0,0], coef[0,1], color='r')
plt.arrow(0,0, coef[1,0], coef[1,1], color='r')
plt.arrow(0,0, coef[2,0], coef[2,1], color='r')
plt.arrow(0,0, coef[3,0], coef[3,1], color='r')
plt.arrow(0,0, coef[4,0], coef[4,1], color='r')
# %%
