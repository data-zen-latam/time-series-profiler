import numpy as np
import pandas as pd
import plotnine as pn
from tslearn.clustering import TimeSeriesKMeans

url = 'https://raw.githubusercontent.com/Mcompetitions/M4-methods/refs/heads/master/Dataset/Train/Yearly-train.csv'
url_metadata = 'https://raw.githubusercontent.com/Mcompetitions/M4-methods/refs/heads/master/Dataset/M4-info.csv'

metadata_df = pd.read_csv(url_metadata)
df = pd.read_csv(url, index_col=0)


df_desc = df.T.describe().T

df_desc.sort_values('count')

time_series = df.T.reset_index(drop=True)
time_series.columns = time_series.columns.to_list()

time_series['time_step'] = list(range(0, len(time_series)))

time_series = time_series.melt(id_vars='time_step')

time_series['value_std'] = time_series.groupby('variable')['value'].transform(
    lambda x: (x - x.mean())/x.std()
)

time_series['value_norm'] = time_series.groupby('variable')['value'].transform(
    lambda x: (x - x.min())/(x.max() - x.min())
)

time_series = time_series.merge(
    metadata_df,
    how='left',
    left_on=['variable'],
    right_on=['M4id']
)

(
    pn.ggplot()
    + pn.geom_line(pn.aes(x='time_step', y = 'value_norm', group='variable'), data=time_series)
)


(
    pn.ggplot()
    + pn.geom_line(pn.aes(x='time_step', y = 'value_norm', group='variable'), data=time_series)
    + pn.facet_wrap('~category')
)

# model = KMeans(n_clusters=8)
model = TimeSeriesKMeans(n_clusters=15, metric='dtw', n_jobs=-1)


series_lens = time_series.groupby(['variable'])['value'].agg(lambda x : x.notna().sum()).reset_index()

# unique_lens = series_lens['value'].value_counts().reset_index().query('count>10')['value']
# unique_lens.sort_values()
# series_lens.query('value in @unique_lens')

q_75 = np.quantile(series_lens['value'], 0.75)
q_25 = np.quantile(series_lens['value'], 0.25)

iqr = q_75 - q_25
max_len, min_len = (q_75 + 1.5*iqr, q_25 - 1.5*iqr)

mask = (series_lens['value'] < max_len) & (series_lens['value'] > min_len)
filter_series = series_lens[mask]['variable'].unique()

max_len_value = series_lens[mask]['value'].max()

(
    pn.ggplot()
    + pn.geom_histogram(pn.aes(x='value'), data=series_lens[mask])
)

input_matrix = time_series.query('variable in @filter_series').pivot(
    index=['variable'],
    columns=['time_step'],
    values='value_std'
)
input_matrix = input_matrix.iloc[:, 0: max_len_value + 1]

# input_matrix = input_matrix.fillna(0)

labels = model.fit_predict(input_matrix.to_numpy())

cluster_centers = model.cluster_centers_

input_matrix['cluster'] = labels

input_matrix.describe()

series_cluster = input_matrix['cluster'].reset_index()

cluster_result = time_series.query('variable in @filter_series').merge(series_cluster, how='left', on='variable')

cluster_result = cluster_result.query('time_step <= @max_len_value')#.fillna(0)

pd.DataFrame(cluster_centers)

cluster_centers_df = pd.DataFrame(cluster_centers[:,:,0]).reset_index().melt(id_vars='index').rename({'variable':'time_step', 'index':'cluster'}, axis=1)

cluster_centers_df['time_step'] = cluster_centers_df['time_step'].astype(int)
(
    pn.ggplot()
    + pn.geom_line(pn.aes(x='time_step', y='value_std', group='variable'), data = cluster_result, color='gray')
    + pn.geom_line(pn.aes(x='time_step', y='value'), data = cluster_centers_df, color='red')
    + pn.facet_wrap('cluster')
)

(
    pn.ggplot()
    + pn.geom_line(pn.aes(x='time_step', y='value'), data = cluster_centers_df, color='red')
    + pn.facet_wrap('cluster')
)

