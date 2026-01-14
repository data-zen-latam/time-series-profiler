# Time Series Profiler

Classify time series by (forecastability)[https://arxiv.org/html/2507.13556v1] as defined in AWS' "Time Series Forecastability Measures". Also gives statistics on data quality, and visualize results in a 3D scatterplot, treating data quality as one dimension, frequency complexity as another one, and chaotic behavior as the final one. This repository uses uv for virtual environment and dependency management.

## Contents

- Overview of goals and metrics
- Planned repository structure and workflow
- uv setup and run instructions
- Input format, pipeline, and outputs
- Extensions and data transparency notes

---

## Overview

The goal is to ingest a `.csv` of time series, compute complexity, chaos, and data quality metrics per series, and produce a 3D scatterplot:

- X-axis: complexity (e.g., spectral entropy, dominant frequency ratio)
- Y-axis: data quality (aggregated profiling score)
- Z-axis: chaos metric (e.g., largest Lyapunov exponent)

All results can optionally be exported as a `.csv` for downstream analysis. 2D projections (e.g., complexity vs. quality) can be provided as supplementary views.

---

## Repository Structure

- **/src**
	- [src/main.py](src/main.py): entry point and pipeline orchestration
	- [src/complexity.py](src/complexity.py): complexity metrics (spectral entropy, dominant frequency ratio)
	- [src/chaos.py](src/chaos.py): chaos metrics (e.g., largest Lyapunov exponent via delay embedding)
	- [src/data_quality.py](src/data_quality.py): profiling metrics (missingness, outliers, distribution stats)
	- [src/visualization.py](src/visualization.py): 3D scatterplot rendering (plotly/matplotlib mplot3d) and 2D projections
	- [src/utils.py](src/utils.py): helpers (CSV parsing, grouping, windowing)
- **/tests**
	- Unit tests per module (e.g., `test_complexity.py`, `test_data_quality.py`)
- **/examples**
	- Example scripts that run the pipeline using data in `data/raw/` (data files are not tracked; download locally)
- [README.md](README.md)
	- Usage, design, and metric explanations

Note: This structure is planned. Files will be added as modules are implemented.

---

## Setup (uv)

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Create and sync a local virtual environment
uv venv .venv
uv sync

# Verify environment works
uv run python -V
```

To add dependencies as development progresses:

```bash
# Examples (adjust as modules land)
uv add numpy pandas scipy matplotlib seaborn plotly
uv add ydata-profiling
uv add scikit-learn
uv sync
```

---

## Input Format

**Required** CSV columns:

- **timestamp**: time index (parseable as datetime or numeric)
- **value**: observed value

Example (single series):

```csv
timestamp,value
2024-01-01,0.12
2024-01-02,0.15
2024-01-03,0.18
```

---

## Complexity Metrics (Distinct from Chaos)

- **Spectral Predictability**
	- Compute power spectrum via FFT
	- Quantify regularity via spectral entropy or dominant frequency ratio

## Chaos Metrics

- **Largest Lyapunov Exponent**
	- Reconstruct phase space using delay embedding
	- Estimate the exponent with standard algorithms
	- Warn or fallback for short/sparse series

---

## Data Quality Profiling

- Use a profiling library (e.g., ydata-profiling) or custom aggregations
- Extract per-series metrics:
	- Missing value ratio
	- Outlier count ratio
	- Series length > 100 obs
	- Distribution stats (mean, std, skewness, kurtosis)
    - Unique value ratio
- Aggregate into a single **data quality score** (e.g., weighted sum or PCA)

---

## Pipeline

1. **Input**: Read `.csv` with columns (`timestamp`, `value`)
2. **Preprocessing**: Parse and validate types
3. **Metric Calculation**: Compute complexity, chaos, and data quality (kept separate)
4. **Visualization**: Create 3D scatterplot (x: complexity, y: data quality, z: chaos). Provide 2D projections optionally.
5. **Export**: Optional `.csv` with all metrics

---

## Outputs

- **3D scatterplot**: Interactive HTML + JavaScript (via Plotly), browser-viewable with rotation, zoom, hover tooltips
- Optional 2D projections (static or interactive) for specific pairs (e.g., complexity vs. quality)
- Optional metrics table as `.csv`

---

## Run (Planned CLI)

The following interface will be implemented in [src/main.py](src/main.py):

```bash
# Basic run
uv run python src/main.py \
	--input data/raw/sales_train_validation.csv \
	--output reports/metrics.csv \
	--plot3d reports/scatter3d.html

# Options (subject to change during development)
uv run python src/main.py \
	--window-size 256 \
	--embedding-dim 5 \
	--delay 2 \
	--projection xy  # show 2D projection (xy, xz, yz)
	--interactive
```

---

## Extensions

- Sliding window analysis for non-stationary series
- Automated model selection or anomaly detection integration
- Alternate frontends: CLI and simple web UI

---

## Data Transparency

- Clearly mark series with estimated or unreliable metrics (e.g., short/sparse)
- Include a legend and data source summary in the plot

---

## Development

- **Testing**: add unit tests under [tests](tests). Example commands:

```bash
uv run pytest -q
```

- **Format/Lint**: choose tools (e.g., black, ruff) and add them via `uv add` as needed.

- **Data**: `data/raw/` is preserved; other data folders are ignored by git.

---

## Example Datasets

- **M5 Forecasting Competition**: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data
	- Requires a Kaggle profile and joining/subscribing to the competition to access data.

### Setup Kaggle Credentials

1. Visit https://www.kaggle.com/settings/account and click "Create New API Token"
2. Copy your username and key to `.env`:

```bash
KAGGLE_USERNAME=your_username
KAGGLE_API_TOKEN=your_api_key
```

### Download M5 Data

```bash
# Load environment variables
source .env

# Accept terms at: https://www.kaggle.com/competitions/m5-forecasting-accuracy
# Then download directly to data/raw/
cd data/raw
kaggle competitions download -c m5-forecasting-accuracy -p .
unzip -q m5-forecasting-accuracy.zip
# Keep only CSVs
find . -maxdepth 1 -type f ! -name '*.csv' -delete
cd ../..
```

### M5 Dataset Description (from Kaggle)

The M5 competition uses hierarchical Walmart sales data (California, Texas, Wisconsin) with item, department, category, and store details, plus explanatory variables (prices, promos, events). You forecast daily sales for 28 days. This complements the uncertainty-estimation track.

---

## License

MIT (see [LICENSE](LICENSE)).
