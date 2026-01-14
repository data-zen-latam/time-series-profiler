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
- Z-axis: chaotic behavior metric (e.g., largest Lyapunov exponent)

All results can optionally be exported as a `.csv` for downstream analysis. 2D projections (e.g., complexity vs. quality) can be provided as supplementary views.

---

## Repository Structure

- **/src**
	- [src/main.py](src/main.py): entry point and pipeline orchestration
	- [src/complexity.py](src/complexity.py): complexity metrics (spectral entropy, dominant frequency ratio)
	- [src/chaos.py](src/chaos.py): chaotic behavior metrics (e.g., largest Lyapunov exponent via delay embedding)
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

## Implemented Metrics

### Complexity: Spectral Predictability

Implemented in [src/complexity.py](src/complexity.py#L9):

- **Method**: LOESS detrending (removes trend) → Welch's method for PSD estimation (averaged periodograms with Hann windowing) → Shannon entropy → predictability score
- **Output**: Float in [0, 1]
  - High values (0.7-1.0): Regular, concentrated frequency spectrum (predictable)
  - Low values (0-0.3): Flat, dispersed spectrum (unpredictable/complex)
- **Formula**: Ω = 1 - H / log(m) where H is spectral entropy, m is number of frequency bins
- **Interpretation**: 
  - Pure sine wave: high predictability (~0.5-0.7)
  - White noise: low predictability (~0)
  - Multi-frequency signals: intermediate predictability based on spectral concentration

### Chaos: Largest Lyapunov Exponent

Implemented in [src/chaos.py](src/chaos.py#L56):

- **Method**: Time-delay embedding with auto-estimated delay (via ACF on detrended data) → nearest neighbor tracking → divergence estimation over fixed evolution steps → sigmoid transformation
- **Output**: Float in [0, 1] (chaos score)
  - < 0.5: Stable, regular behavior
  - = 0.5: Neutral (λ = 0)
  - > 0.5: Chaotic, unpredictable behavior
- **Delay Estimation**: Automatic via autocorrelation, finds first lag where ACF < 0.3 or local minimum
- **Data Requirements**: Minimum 100×m points (m = embedding dimension, default 5)
- **Fallback**: Returns NaN if insufficient data; main.py switches to 2D plot (complexity vs quality)

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

## Run (CLI)

The CLI is implemented in [src/main.py](src/main.py):

```bash
# Basic run (single series CSV)
uv run python -m src.main \
	--input data/raw/sales_train_validation.csv \
	--output reports/metrics.csv \
	--plot3d reports/scatter3d.html

# Full options
uv run python -m src.main \
	--input data/raw/series.csv \
	--output reports/metrics.csv \
	--plot3d reports/scatter3d.html \
	--embedding-dim 5 \
	--window-size 256
```

**Behavior**:
- Computes complexity (spectral entropy), chaos (Lyapunov exponent), and data quality metrics
- Exports metrics to CSV
- Creates 3D scatterplot (complexity vs quality vs chaos)
- If chaos unavailable (NaN), falls back to 2D plot (complexity vs quality) with warning logged

---

## Utilities

[src/utils.py](src/utils.py) provides:

- **`load_csv(file_path)`**: Load CSV file
- **`detrend_series(values)`**: STL decomposition to extract residuals (trend + seasonality removed)

## Development Status

**Completed**:
- ✅ Spectral entropy with STL detrending
- ✅ Largest Lyapunov exponent with auto delay estimation
- ✅ 3D scatterplot with 2D fallback
- ✅ CLI pipeline in main.py

**In Progress**:
- ⊙ Data quality metrics (quality.py stubs)
- ⊙ Visualization functions (visualization.py stubs)
- ⊙ Unit tests

**Planned**:
- Sliding window analysis
- Automated anomaly detection
- Web UI

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

## Validation with Synthetic Data

Test the metrics on synthetic time series with increasing forecasting difficulty:

```bash
# Generate synthetic series and compute metrics
uv run python examples/synthetic_validation.py
```

This creates interactive plots in `examples/synthetic_plots/`:
1. Pure sine wave (simplest)
2. Multi-frequency wave
3. Noisy multi-frequency wave
4. Lorenz system (chaotic)
5. White noise (most complex)

View the plots:

```bash
# Option 1: Open directly (browser)
open examples/synthetic_plots/1_pure_sine.html

# Option 2: Serve locally and view
python -m http.server 8000
# Then navigate to http://localhost:8000/examples/synthetic_plots/
```

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
