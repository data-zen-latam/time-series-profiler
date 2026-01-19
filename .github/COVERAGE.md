# Coverage Report

**Last Updated:** 2026-01-16  
**Overall Coverage:** 74%

## Module Breakdown

| Module | Coverage | Status | Notes |
|--------|----------|--------|-------|
| `src/data_quality.py` | **100%** | ✅ | Fully tested - missing value, outlier, stationarity checks |
| `src/complexity.py` | **95%** | ✅ | Nearly complete - one edge case untested |
| `src/chaos.py` | **87%** | ⚠️ | Good coverage - 18 lines untested (mostly error handling) |
| `src/__init__.py` | **100%** | ✅ | Module exports fully covered |
| `src/utils.py` | **75%** | ⚠️ | Utility functions - mainly logging paths untested |
| `src/visualization.py` | **0%** | ❌ | Not tested - requires plotting environment |

## Test Statistics

- **Total Tests:** 82
- **Pass Rate:** 100%
- **Warnings:** 2 (harmless - divide by zero in edge cases)
- **Execution Time:** ~27-30 seconds

### Tests by Module

| Module | Tests | Classes | Focus |
|--------|-------|---------|-------|
| `test_chaos.py` | 18 | 4 | Lyapunov exponent, embedding dimension, entropy |
| `test_complexity.py` | 32 | 6 | Spectral entropy, Hurst exponent, fractal dimension, ApEn, SampEn |
| `test_data_quality.py` | 32 | 5 | Missing values, outliers, stationarity, autoregression, unique ratio |

## GitHub Actions Mapping

### Intelligent Triggers

The CI/CD pipeline uses **path-based triggers** to run only relevant tests:

1. **test-chaos.yml** - Triggers on:
   - Changes to `src/chaos.py`
   - Changes to `tests/test_chaos.py`
   - Changes to shared fixtures (`tests/conftest.py`)
   - Workflow file changes

2. **test-complexity.yml** - Triggers on:
   - Changes to `src/complexity.py`
   - Changes to `tests/test_complexity.py`
   - Changes to shared fixtures (`tests/conftest.py`)
   - Workflow file changes

3. **test-data-quality.yml** - Triggers on:
   - Changes to `src/data_quality.py`
   - Changes to `tests/test_data_quality.py`
   - Changes to shared fixtures (`tests/conftest.py`)
   - Workflow file changes

4. **full-test-suite.yml** - Triggers on:
   - Any push to `main` or `develop`
   - Any pull request to `main` or `develop`
   - Daily schedule (2 AM UTC)
   - Runs complete test suite across Python 3.10, 3.11, 3.12

## Coverage Goals

- **Short-term:** Maintain 74%+ coverage
- **Medium-term:** Achieve 85%+ coverage (add visualization tests)
- **Long-term:** Reach 95%+ coverage (edge cases and error paths)

## Uncovered Code

### `src/chaos.py` (13% uncovered)
- Lines 36, 42, 54-60: Exception handling paths
- Lines 83, 96: Error guards in specialized functions
- Lines 130, 135, 208, 239, 245, 269, 278, 295, 302: Edge case handling

### `src/complexity.py` (5% uncovered)
- Line 29: One condition branch in error handling

### `src/utils.py` (25% uncovered)
- Lines 30, 38-40, 53, 66, 80: Mainly logging paths and optional features

### `src/visualization.py` (100% uncovered)
- Requires plotting environment and visual validation
- Not suitable for automated unit testing
- Consider acceptance tests or manual validation

## How to Generate Coverage

```bash
# Full coverage report
uv run pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

# Module-specific coverage
uv run pytest tests/test_chaos.py --cov=src.chaos --cov-report=term-missing
uv run pytest tests/test_complexity.py --cov=src.complexity --cov-report=term-missing
uv run pytest tests/test_data_quality.py --cov=src.data_quality --cov-report=term-missing

# View HTML report
open htmlcov/index.html
```
