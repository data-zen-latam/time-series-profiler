# Synthetic Data Validation Example

This example demonstrates the time-series profiler metrics on five synthetic time series with known properties and complexity characteristics.

## Overview

The validation script generates five synthetic series with increasing complexity and chaotic behavior:

1. **Pure Sine Wave**: Simplest, most predictable series
   - Single frequency component
   - Expected: Low complexity, low chaos

2. **Multi-Frequency Wave**: Sum of three sine waves
   - Multiple periodic components
   - Expected: Medium complexity, low chaos

3. **Noisy Multi-Frequency**: Multi-frequency signal + small stochastic component
   - Signal with additive noise (SNR ≈ 10)
   - Expected: Medium-high complexity, low chaos

4. **Lorenz System**: Chaotic attractor trajectory
   - Deterministic but chaotic dynamics
   - Expected: High complexity, high chaos

5. **White Noise**: Completely random signal
   - Maximum unpredictability, flat frequency spectrum
   - Expected: Highest complexity, variable chaos

## Running the Validation

```bash
uv run python examples/synthetic_data_validation/validate.py
```

## Output

### Console Output

Displays a table with computed metrics:

```
======================================================================
Synthetic Time Series Validation
======================================================================

Series                    Complexity      Chaos          
----------------------------------------------------------------------
1. Pure Sine               X.XXXX          X.XXXX
2. Multi-Frequency        X.XXXX          X.XXXX
3. Noisy Multi-Freq       X.XXXX          X.XXXX
4. Lorenz System          X.XXXX          X.XXXX
5. White Noise            X.XXXX          X.XXXX
----------------------------------------------------------------------
```

### Visualization

Generates an interactive HTML file: `output/series_plots_and_psds.html`

Contains:
- **Left column**: Detrended time series for each synthetic series
- **Right column**: Power Spectral Density (PSD) plots in log scale
- Hover tooltips with spectral entropy and predictability metrics

## Metrics Explanation

### Complexity (Spectral Predictability)

**Range**: [0, 1]

- **High values (0.7-1.0)**: Regular, concentrated frequency spectrum (predictable)
- **Low values (0-0.3)**: Flat, dispersed spectrum (unpredictable/complex)

Computed using:
1. LOESS detrending to remove trend component
2. Welch's method for PSD estimation (averaged periodograms)
3. Shannon entropy of the normalized power distribution
4. Formula: C = H / log(m), where H is spectral entropy and m is the number of frequency bins

### Chaos (Chaotic Behavior)

**Range**: [0, 1]

- **< 0.5**: Stable, regular behavior
- **= 0.5**: Neutral (zero Lyapunov exponent)
- **> 0.5**: Chaotic, divergent trajectories

Computed using:
1. Automatic delay estimation via autocorrelation
2. Time-delay embedding (dimension m = 5)
3. Largest Lyapunov exponent via nearest neighbor divergence
4. Sigmoid transformation to [0, 1] range

## Expected Behavior

Ideally, metrics should show:

- Complexity increasing from series 1 → 5
- Chaos increasing from series 1 → 5
- Series 4 (Lorenz) showing notably high chaos (>0.5)
- Series 1 (Pure sine) showing low values for both metrics (<0.3)

## Viewing Results

To view the generated HTML plot in a browser:

```bash
# Option 1: Using Python's built-in server
python -m http.server 8000 --directory examples/synthetic_data_validation/output

# Option 2: Open directly in a browser
open examples/synthetic_data_validation/output/series_plots_and_psds.html
```

Then navigate to: `http://localhost:8000/series_plots_and_psds.html`
