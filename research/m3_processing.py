#%%
import pandas as pd
from scipy.stats import entropy
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

mu = 0
std = 1
series['w_noise'] = np.random.normal(mu, std, size=series.get('N1402').shape[0])
series['sine'] = np.sin(np.arange(0,series.get('N1402').shape[0]))
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
#%%

from operator import itemgetter

x = data['month']['Series'].to_numpy()
x = np.concat([x, ['w_noise', 'sine']])

a = itemgetter(*x)(series)

a = np.array(a)
df = pd.DataFrame(a).T
df.columns = x
df['index'] = np.arange(0, df.shape[0])

#%%
result = result[np.isin(result['series'], x)]

#%%


aux = df.melt(id_vars='index', var_name='series')

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
aux['score_entropy'] = 1 - (aux['entropy'] / entropy(np.ones(df.shape[0]) / df.shape[0]))

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
# aux['score_entropy'] = 1 - (aux['entropy'] / entropy(np.ones(df.shape[0]) / df.shape[0]))
# switch for fourier entropy

aux['predictability_score'] = aux.loc[:, aux.columns.str.contains('score')].mean(axis=1)

scores_ = pd.concat([scores_, aux], axis=1)

#%%
import seaborn as sns

sns.scatterplot(scores_, x='completeness_score', y='predictability_score')

#%%
mask = ((scores_['predictability_score'] < 0.2) & (scores_['completeness_score'] <0.82))

scores_[mask].T
scores_.iloc[-2:,:].T
#%%
x = series.get('N2755')
sns.lineplot(x=np.arange(0, df.shape[0]), y=x)

# x = series.get('N1850')
x = series.get('N2573')
sns.lineplot(x=np.arange(0, df.shape[0]), y=x)
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
