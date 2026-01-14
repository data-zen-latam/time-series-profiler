"""Validate complexity and chaotic behavior metrics on synthetic time series.

Generates five segments with increasing forecasting difficulty:
1. Pure sine wave (simple, predictable)
2. Multi-frequency wave (periodic but complex)
3. Noisy multi-frequency wave (complex + stochastic)
4. Lorenz system trajectory (chaotic)
5. White noise (random, unpredictable)

Produces:
- Validation metrics table (printed to stdout)
- Combined HTML visualization with time series, PSDs, and metrics
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import welch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chaos import largest_lyapunov_exponent
from src.complexity import spectral_entropy
from src.utils import detrend_series


def generate_sine_wave(n=1000, freq=0.1):
    """Pure sine wave - simplest, most predictable."""
    t = np.arange(n)
    return np.sin(2 * np.pi * freq * t)


def generate_multi_frequency(n=1000, freqs=[0.05, 0.1, 0.15]):
    """Multi-frequency wave - more complex spectrum but still periodic."""
    t = np.arange(n)
    signal = np.zeros(n)
    for freq in freqs:
        signal += np.sin(2 * np.pi * freq * t)
    return signal / len(freqs)


def generate_noisy_multi_frequency(n=1000, freqs=[0.05, 0.1, 0.15], noise_std=0.05):
    """Noisy multi-frequency - complex + small stochastic component."""
    signal = generate_multi_frequency(n, freqs)
    noise = np.random.normal(0, noise_std, n)
    return signal + noise


def generate_lorenz(n=1000, dt=0.02, sigma=15, rho=28, beta=8/3):
    """Lorenz system - highly chaotic attractor.
    
    Args:
        n: number of samples
        dt: time step - larger dt = more chaotic behavior
        sigma: parameter
        rho: parameter - larger rho = more chaotic
        beta: parameter
    """
    # Initial conditions
    x, y, z = 0.0, 1.0, 1.05
    
    trajectory = np.zeros(n)
    
    for i in range(n):
        # Lorenz equations
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        
        x += dx * dt
        y += dy * dt
        z += dz * dt
        
        trajectory[i] = x  # Use x component as time series
    
    return trajectory


def generate_white_noise(n=1000):
    """White noise - maximum unpredictability."""
    return np.random.normal(0, 1, n)


def main():
    """Generate synthetic series, compute metrics, and create combined visualization."""
    parser = argparse.ArgumentParser(description="Validate complexity and chaos metrics on synthetic data")
    args = parser.parse_args()
    
    np.random.seed(42)  # Reproducibility
    
    # Generate all segments
    segments = {
        "1. Pure Sine": generate_sine_wave(),
        "2. Multi-Frequency": generate_multi_frequency(),
        "3. Noisy Multi-Freq": generate_noisy_multi_frequency(),
        "4. Lorenz System": generate_lorenz(),
        "5. White Noise": generate_white_noise(),
    }
    
    print("=" * 70)
    print("Synthetic Time Series Validation")
    print("=" * 70)
    print()
    print(f"{'Series':<25} {'Complexity':<15} {'Chaotic Behavior':<15}")
    print("-" * 70)
    
    results = []
    series_list = []
    
    for name, series in segments.items():
        # Compute metrics
        complexity = spectral_entropy(series)
        chaos = largest_lyapunov_exponent(series)
        
        results.append({
            "name": name,
            "complexity": complexity,
            "chaos": chaos,
            "series": series
        })
        
        series_list.append((name, series))
        print(f"{name:<25} {complexity:>7.4f}        {chaos:>7.4f}")
    
    print("-" * 70)
    print()
    
    # Validate trends
    complexities = [r["complexity"] for r in results if not np.isnan(r["complexity"])]
    chaos_scores = [r["chaos"] for r in results if not np.isnan(r["chaos"])]
    
    print("Validation:")
    if len(complexities) >= 2:
        complexity_trend = "increasing" if complexities[-1] > complexities[0] else "not increasing"
        print(f"  ✓ Complexity trend: {complexity_trend}")
    
    if len(chaos_scores) >= 2:
        chaos_trend = "increasing" if chaos_scores[-1] > chaos_scores[0] else "not increasing"
        print(f"  ✓ Chaos trend: {chaos_trend}")
    
    print()
    print("Expected behavior:")
    print("  - Complexity and chaotic behaviour scores should increase from series (1) to (5)")
    print("  - (4) Lorenz should show high chaos score (>0.5)")
    print("  - (1) Pure sine should show low complexity and chaos (<0.3)")
    print()
    
    # Create combined visualization with time series, PSDs, and metrics
    titles = []
    for name, _ in series_list:
        titles.append(f'{name.split(".")[1].strip()}<br>Detrended')
        titles.append(f'{name.split(".")[1].strip()}<br>PSD')
    
    fig = make_subplots(
        rows=5, cols=2,
        subplot_titles=titles,
        vertical_spacing=0.08,
        horizontal_spacing=0.12
    )
    
    for idx, (name, series) in enumerate(series_list):
        row = idx + 1
        
        # Detrend
        series_detrended = detrend_series(series)
        
        # Compute PSD
        freqs, psd = welch(series_detrended, nperseg=min(256, len(series)))
        
        # Normalize to probabilities
        p = psd / psd.sum()
        
        # Compute entropy
        entropy = -np.sum(p * np.log(p + np.finfo(float).eps))
        max_entropy = np.log(len(psd))
        complexity = entropy / max_entropy
        
        # Time series plot (left column)
        fig.add_trace(
            go.Scatter(y=series_detrended, mode='lines', line=dict(width=1),
                       name=f'{name} - Time Series', showlegend=False),
            row=row, col=1
        )
        fig.update_yaxes(title_text="Amplitude", row=row, col=1)
        
        # PSD plot (right column) - log scale
        fig.add_trace(
            go.Scatter(x=freqs, y=psd, mode='lines', line=dict(width=1.5),
                       name=f'C={complexity:.3f}', 
                       showlegend=False),
            row=row, col=2
        )
        fig.update_yaxes(title_text="Power", type="log", row=row, col=2)
        fig.update_xaxes(title_text="Frequency", row=row, col=2)
    
    # Update layout
    fig.update_xaxes(title_text="Sample", row=5, col=1)
    fig.update_layout(
        height=1400,
        title_text="Synthetic Time Series Analysis: Detrended Series and Power Spectral Density",
        showlegend=False
    )
    
    output_file = Path(__file__).parent / 'output' / 'series_plots_and_psds.html'
    output_file.parent.mkdir(exist_ok=True)
    fig.write_html(output_file)
    print(f"Visualization saved to: {output_file.absolute()}")
    print()


if __name__ == "__main__":
    main()
