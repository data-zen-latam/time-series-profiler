# Time Series Profiler

Classify time series by (forecastability)[https://arxiv.org/html/2507.13556v1] as defined in AWS' "Time Series Forecastability Measures". Also gives statistics on data quality, and visualize results in a 3D scatterplot, treating data quality as one dimension, frequency complexity as another one, and chaotic behavior as the final one. This repository uses uv for virtual environment and dependency management.

---

## Overview

The goal is to ingest a `.csv` of time series, compute complexity, chaos, and data quality metrics per series, and produce a 3D scatterplot:

- X-axis: data quality (aggregated profiling score)
- Y-axis: complexity (spectral entropy)
- Z-axis: chaotic behavior (largest Lyapunov exponent)

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
uv add scikit-learn
uv sync
```

## Implemented Metrics

### Complexity: Spectral Predictability

Implemented in [src/complexity.py](src/complexity.py):

- **Method**: LOESS detrending (removes trend) → Welch's method for PSD estimation (averaged periodograms with Hann windowing) → Shannon entropy → normalized entropy
- **Output**: Float in [0, 1]
	- Higher values: More dispersed spectrum (higher complexity)
	- Lower values: Concentrated spectrum (lower complexity)
- **Formula**: C = H / log(m) where H is spectral entropy, m is the number of frequency bins
- **Interpretation**: 
	- Pure sine wave: low complexity
	- White noise: highest complexity
	- Multi-frequency signals: medium complexity

### Chaos: Largest Lyapunov Exponent

Implemented in [src/chaos.py](src/chaos.py):

- **Method**: Time-delay embedding with auto-estimated delay (ACF on detrended data) → nearest neighbor tracking → divergence estimation over fixed evolution steps
- **Output**: Float chaos score (scaled λa, guard for constant/degenerate series)
- **Delay Estimation**: Automatic via autocorrelation; first lag below a threshold or local minimum
- **Embedding Dimension**: Auto-estimated via Cao's method (E1/E2 plateau detection)
- **Data Requirements**: Accounts for short/degenerate inputs; returns NaN when unreliable

---

## Data Quality Profiling

Custom aggregations implemented:
- Missing value ratio
- Outlier count ratio (z-score based; guarded against zero variance)
- Distribution stats (mean, std, skewness, kurtosis)
- Unique value ratio
- Aggregation: (1 - missing) * (1 - outlier) * unique_ratio → clamped to [0, 1]

---

## Outputs

- **3D scatterplot**: Interactive HTML + JavaScript (via Plotly), browser-viewable with rotation, zoom, hover tooltips
- Optional 2D projections (static or interactive) for specific pairs (e.g., complexity vs. quality)

---

## Quick Start (Examples)

```bash
# Activate virtual environment
source .venv/bin/activate

# 1) Synthetic validation table + time series & PSD plots
python examples/synthetic_data_validation/validate.py

# 2) 3D metric space (Data Quality vs Complexity vs Chaos)
python examples/synthetic_data_validation/plot_metric_space_3d.py

# Optional: serve outputs in browser
python -m http.server 8000
# Visit: http://localhost:8000/examples/synthetic_data_validation/output/
```

---

## Utilities

[src/utils.py](src/utils.py) provides:

- **`detrend_series(values)`**: LOESS detrending to remove smooth trends while preserving oscillatory components
- **`load_csv(file_path)`**: Load CSV into DataFrame (basic wrapper)

## Testing

Run the test suite:

```bash
# All tests
uv run pytest -v

# Specific module
uv run pytest tests/test_chaos.py -v

# With coverage
uv run pytest --cov=src --cov-report=html
```

**Test Statistics:**
- 82 tests across 3 modules (chaos, complexity, data_quality)
- 74% overall coverage
- 100% pass rate
- CI/CD via GitHub Actions (triggers on main branch changes)

---

## Validation with Synthetic Data

Two example scripts demonstrate the metrics and visualizations:

- `examples/synthetic_data_validation/validate.py`
	- Prints a metrics table (Complexity, Chaos) for six synthetic series
	- Saves a combined HTML with detrended series and PSDs
- `examples/synthetic_data_validation/plot_metric_space_3d.py`
	- Plots each series as a point in 3D (Data Quality, Complexity, Chaos)
	- Uses a colorblind-friendly Okabe–Ito palette and visible grid for depth

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
