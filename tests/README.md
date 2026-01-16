# Test Suite

Comprehensive unit tests for the time-series-profiler using pytest.

## Quick Start

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_chaos.py

# Run specific test class
uv run pytest tests/test_chaos.py::TestLargestLyapunovExponent

# Run specific test method
uv run pytest tests/test_chaos.py::TestLargestLyapunovExponent::test_constant_series_returns_zero

# Run tests matching a pattern
uv run pytest -k "constant"

# Show test coverage
uv run pytest --cov=src --cov-report=html
```

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures (test data)
├── test_chaos.py           # Tests for chaos.py (Lyapunov exponent)
├── test_complexity.py      # Tests for complexity.py (spectral entropy)
├── test_data_quality.py    # Tests for data_quality.py (quality metrics)
└── README.md               # This file
```

## Test Organization

### conftest.py - Shared Fixtures

Fixtures are reusable test data that pytest automatically injects into tests:

```python
@pytest.fixture
def lorenz_series():
    """Expensive Lorenz attractor generation - created once, reused everywhere."""
    # ... generate data ...
    return np.array(xs)

# Use in any test file:
def test_lorenz_positive(lorenz_series):  # ← automatically injected
    lyap = largest_lyapunov_exponent(lorenz_series)
    assert lyap > 0.5
```

**Available fixtures:**
- `constant_series` - All ones (zero variance)
- `sine_series` - Pure sine wave (periodic)
- `multi_frequency` - Multiple sine waves (moderate complexity)
- `white_noise` - Random noise (high complexity)
- `lorenz_series` - Lorenz attractor (chaotic)
- `perfect_series` - Perfect quality data
- `normal_distribution` - Standard normal distribution
- `series_with_outliers` - Data with extreme values

### test_chaos.py - Chaotic Behavior Tests

Tests for Lyapunov exponent calculation and Cao's method:

**Edge cases covered:**
- Constant series → λ = 0
- Pure sine → λ ≈ 0
- Lorenz attractor → λ > 0
- NaN/inf handling
- Very short series
- Auto-parameter estimation

**Example parametrized test:**
```python
@pytest.mark.parametrize("value", [0.0, 1.0, 42.0, -5.0, 1e6])
def test_constant_series_returns_zero(value):
    # Creates 5 separate tests, one for each value
    series = np.full(500, value)
    assert largest_lyapunov_exponent(series) == pytest.approx(0.0)
```

### test_complexity.py - Spectral Entropy Tests

Tests for complexity metrics via spectral entropy:

**Edge cases covered:**
- Constant series → entropy = 0
- Pure sine → low entropy
- White noise → high entropy
- Multiple frequencies → moderate entropy
- NaN/inf handling
- Uniform power spectral density

**Example with fixtures:**
```python
def test_white_noise_high_entropy(white_noise):  # ← fixture
    entropy = spectral_entropy(white_noise)
    assert entropy > 0.7  # Should be close to 1.0
```

### test_data_quality.py - Data Quality Tests

Tests for quality metrics and aggregation:

**Metrics tested:**
- `missing_value_ratio()` - NaN and inf detection
- `outlier_count_ratio()` - Z-score based outliers
- `unique_value_ratio()` - Uniqueness measure
- `aggregate_quality_score()` - Combined quality score

**Edge cases covered:**
- Perfect data
- All missing values
- Constant series
- Various outlier thresholds
- Empty series

## Pytest Features Demonstrated

### 1. Fixtures (conftest.py)
Reusable test data setup:
```python
@pytest.fixture
def sine_series():
    t = np.linspace(0, 50*np.pi, 1000)
    return np.sin(t)

def test_example(sine_series):  # Auto-injected
    assert len(sine_series) == 1000
```

### 2. Parametrization
Run same test with different inputs:
```python
@pytest.mark.parametrize("n_total,n_missing,expected", [
    (10, 0, 0.0),
    (10, 5, 0.5),
    (10, 10, 1.0),
])
def test_partial_missing(n_total, n_missing, expected):
    # Creates 3 separate tests
    series = np.ones(n_total)
    series[:n_missing] = np.nan
    assert missing_value_ratio(series) == pytest.approx(expected)
```

### 3. pytest.approx
Float comparison with tolerance:
```python
# Bad - may fail due to floating point errors
assert lyapunov == 0.0

# Good - allows small tolerance
assert lyapunov == pytest.approx(0.0, abs=1e-10)
```

### 4. Test Discovery
Pytest automatically finds tests matching patterns:
- Files: `test_*.py` or `*_test.py`
- Classes: `Test*`
- Functions: `test_*`

## Running Specific Tests

```bash
# Run only chaos tests
uv run pytest tests/test_chaos.py -v

# Run only parametrized tests for constant series
uv run pytest -k "constant" -v

# Run tests and stop at first failure
uv run pytest -x

# Run last failed tests only
uv run pytest --lf

# Show print statements
uv run pytest -s

# Run in parallel (requires pytest-xdist)
uv run pytest -n auto
```

## Writing New Tests

### 1. Add test to appropriate file
```python
# In test_chaos.py
def test_new_feature(sine_series):  # Use fixture if needed
    """Clear description of what this tests."""
    result = new_function(sine_series)
    assert result > 0
```

### 2. Add fixture to conftest.py if reusable
```python
# In conftest.py
@pytest.fixture
def new_test_data():
    """Description of the test data."""
    return generate_data()
```

### 3. Use parametrize for multiple edge cases
```python
@pytest.mark.parametrize("input,expected", [
    (value1, result1),
    (value2, result2),
])
def test_multiple_cases(input, expected):
    assert function(input) == expected
```

## Edge Cases to Test

When adding new functions, always test:

1. **Zero variance** - Constant series
2. **Invalid values** - NaN, inf, -inf
3. **Empty/short** - Empty arrays, very short series
4. **Boundary values** - Min, max, zero
5. **Type validation** - Return types, value ranges
6. **Determinism** - Same input → same output
7. **Expected behaviors** - Known properties (e.g., sine → low chaos)

## Test Output

### Success
```
tests/test_chaos.py::TestLargestLyapunovExponent::test_constant_series_returns_zero[0.0] PASSED
tests/test_chaos.py::TestLargestLyapunovExponent::test_constant_series_returns_zero[1.0] PASSED
tests/test_chaos.py::TestLargestLyapunovExponent::test_lorenz_positive PASSED
```

### Failure
```
tests/test_chaos.py::TestLargestLyapunovExponent::test_sine_near_zero FAILED

=================================== FAILURES ===================================
def test_sine_near_zero(sine_series):
    lyap = largest_lyapunov_exponent(sine_series)
>   assert lyap == pytest.approx(0.0, abs=0.5)
E   assert 0.8 == 0.0 ± 5.0e-01
```

## Test Coverage

Generate coverage report:
```bash
# Install coverage tool
uv add pytest-cov --dev

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Open report
open htmlcov/index.html
```

## Continuous Integration

Tests should run automatically on every commit/PR. See GitHub Actions workflow in `.github/workflows/test.yml` (to be added).

## Dependencies

The test suite requires:
- `pytest` - Testing framework
- `numpy` - Array operations
- `scipy` - Statistical functions (skewness, kurtosis)

Optional:
- `pytest-cov` - Coverage reporting
- `pytest-xdist` - Parallel test execution

## Best Practices

1. **One assertion per test** - Makes failures clear
2. **Descriptive names** - `test_constant_series_returns_zero` not `test_1`
3. **Use fixtures** - Don't repeat setup code
4. **Test edge cases** - Zero, negative, NaN, inf, empty
5. **Use parametrize** - Multiple inputs in one test
6. **Document expected behavior** - Clear docstrings
7. **Fast tests** - Keep under 1 second when possible

## Troubleshooting

**Import errors:**
```bash
# Make sure you're in the project root
cd /path/to/time-series-profiler
uv run pytest
```

**Fixture not found:**
- Check fixture is in `conftest.py`
- Verify parameter name matches fixture name exactly

**Tests pass locally but fail in CI:**
- Check for hardcoded paths
- Verify random seeds are set
- Ensure dependencies are listed in `pyproject.toml`

## Further Learning

- [Pytest Documentation](https://docs.pytest.org/)
- [Effective Python Testing](https://realpython.com/pytest-python-testing/)
- [Parametrize Guide](https://docs.pytest.org/en/stable/how-to/parametrize.html)
- [Fixture Guide](https://docs.pytest.org/en/stable/how-to/fixtures.html)
