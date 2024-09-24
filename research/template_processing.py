#%%

import pandas as pd
import numpy as np
import os
from pathlib import Path

from ts_profiler.config import PROCESSED_DATA_DIR, REPORTS_DIR
from jinja2 import Environment, FileSystemLoader
import plotly.express as px
#%%
PROCESSED_DATA_DIR = Path(os.getcwd()).parents[0] / 'data/processed/'
REPORTS_DIR = Path(os.getcwd()).parents[0] / 'reports/'
data = pd.read_csv(PROCESSED_DATA_DIR / 'features.csv')
df_stats = pd.read_csv(PROCESSED_DATA_DIR / 'descriptive_stats.csv')
df_values = pd.read_csv(PROCESSED_DATA_DIR / 'values_summary.csv')


ts = pd.read_parquet(PROCESSED_DATA_DIR / 'diff_temp_series.parquet.gzip')


#%%
df_stats.info()
df_values.info()

# %%
stats = {
    k : (round(v,2) if k != 'series' else v)
    for k,v in df_stats.to_dict(orient='records')[0].items()
    }
vs = {
    k : (round(v,2) if k not in ['series', 'n', 'n_distinct', 'n_missing', 'n_negative', 'n_zeros'] else v)
    for k,v in df_values.to_dict(orient='records')[0].items()
    }

# %%
fig = px.line(ts, x='date', y='europe').to_html(full_html=False)

#%%
# 2. Create a template Environment
env = Environment(loader=FileSystemLoader(REPORTS_DIR / 'templates'))

# 3. Load the template from the Environment
template = env.get_template('template.html')


# 4. Render the template with variables
html = template.render(
    stats=stats,
    vals=vs,
    fig=fig
)
# %%
with open(REPORTS_DIR / 'html_report_jinja.html', 'w') as f:
    f.write(html)


# %%
