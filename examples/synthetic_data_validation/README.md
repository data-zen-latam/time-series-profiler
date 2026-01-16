# Synthetic Data Validation Example

This example demonstrates the time-series profiler metrics on synthetic time series with known properties and complexity characteristics.

## Overview

The validation scripts generate six synthetic series with increasing complexity and chaotic behavior:

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

6. **Constant (0)**: Degenerate signal (sanity check)
   - No variation, zero entropy
   - Expected: Zero complexity, zero chaos

## Running the Validation

```bash
# Create and sync env with uv (from repo root)
uv venv .venv
uv sync

# Run validation (table + series/PSD plots)
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

### Visualizations

1) Combined detrended series + PSDs: `output/series_plots_and_psds.html`
   - Left: Detrended series
   - Right: Power Spectral Density (log scale)
   - Colorblind-friendly Okabe–Ito palette applied consistently

2) 3D Metric Space (Data Quality vs Complexity vs Chaos): `output/metric_space_3d.html`
   - Points colored via Okabe–Ito palette; size/opacity variations for redundancy
   - Light grey grid (visible) and camera oriented to show origin in lower-left

## Metrics Explanation

### Complexity (Spectral Predictability)

**Range**: [0, 1]

- **Higher values**: Dispersed spectrum (higher complexity)
- **Lower values**: Concentrated spectrum (lower complexity)

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
2. Time-delay embedding with auto-estimated embedding dimension (Cao's method)
3. Largest Lyapunov exponent via nearest neighbor divergence
4. Guarded for constant/degenerate series (returns 0)

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
python -m http.server 8000

# Option 2: Open directly in a browser (paths may vary by OS)
# macOS: open examples/synthetic_data_validation/output/series_plots_and_psds.html
# Linux: xdg-open examples/synthetic_data_validation/output/series_plots_and_psds.html
```

Then navigate to: `http://localhost:8000/examples/synthetic_data_validation/output/`
