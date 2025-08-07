# Time Series Profiler

A Python package for profiling time series data to assess quality and predictability.

## Features

- Time series quality assessment
- Predictability analysis
- Feature extraction and analysis
- Statistical profiling

## Installation

```bash
# Install with uv
uv add ts-profiler

# Install from source
uv pip install -e .

# Install with optional dependencies
uv pip install -e ".[spark]"
```

## Development

```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Run tests
uv run pytest

# Format code
uv run black .

# Lint code  
uv run ruff check .
```

## Usage

```python
from ts_profiler import features, utils

# Your time series analysis code here
```

## Get sample data from M3C

## License

MIT License - see LICENSE file for details.
