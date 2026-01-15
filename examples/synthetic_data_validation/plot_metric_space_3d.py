"""
Generate 3D scatter plot: Data Quality vs Complexity vs Chaotic Behavior.

Each point represents one synthetic time series, positioned according to:
- X-axis: Data Quality Score
- Y-axis: Complexity Score  
- Z-axis: Chaotic Behavior (Largest Lyapunov Exponent)
"""

import numpy as np
import plotly.graph_objects as go
import sys
sys.path.insert(0, '../../src')

from complexity import complexity_score
from chaos import largest_lyapunov_exponent
from data_quality import aggregate_quality_score


def generate_sine_wave(n=1000, freq=0.1):
    """Pure sine wave."""
    t = np.arange(n)
    return np.sin(2 * np.pi * freq * t)


def generate_constant(n=1000, value=0.0):
    """Constant value."""
    return np.full(n, value)


def generate_multi_frequency(n=1000, freqs=None):
    """Multiple frequencies combined."""
    if freqs is None:
        freqs = [0.05, 0.1, 0.25]
    t = np.arange(n)
    signal = np.zeros(n)
    for freq in freqs:
        signal += np.sin(2 * np.pi * freq * t) / len(freqs)
    return signal


def generate_noisy_multi_frequency(n=1000, freqs=None, noise_std=0.3):
    """Multi-frequency with noise."""
    if freqs is None:
        freqs = [0.05, 0.1, 0.25]
    signal = generate_multi_frequency(n, freqs)
    noise = np.random.normal(0, noise_std, n)
    return signal + noise


def generate_lorenz(n=1000, dt=0.02, sigma=15, rho=28, beta=8/3):
    """Lorenz system - chaotic attractor."""
    x, y, z = 0.0, 1.0, 1.05
    trajectory = np.zeros(n)
    
    for i in range(n):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        
        x += dx * dt
        y += dy * dt
        z += dz * dt
        
        trajectory[i] = x
    
    return trajectory


def generate_white_noise(n=1000):
    """White noise."""
    return np.random.normal(0, 1, n)


def main():
    """Generate and plot 3D metric space for all synthetic series."""
    np.random.seed(42)
    
    # Generate all series
    series_dict = {
        "Constant": generate_constant(),
        "Sine": generate_sine_wave(),
        "Multi-Freq": generate_multi_frequency(),
        "Noisy Multi-Freq": generate_noisy_multi_frequency(),
        "Lorenz": generate_lorenz(),
        "White Noise": generate_white_noise(),
    }
    
    # Compute metrics for each series
    names = []
    quality_scores = []
    complexity_scores = []
    chaos_scores = []
    
    print("=" * 80)
    print("Computing 3D Metric Space")
    print("=" * 80)
    print(f"{'Series':<20} {'Data Quality':<18} {'Complexity':<18} {'Chaotic Behavior':<18}")
    print("-" * 80)
    
    for name, series in series_dict.items():
        quality = aggregate_quality_score(series)
        complexity = complexity_score(series)
        chaos = largest_lyapunov_exponent(series)
        
        names.append(name)
        quality_scores.append(quality)
        complexity_scores.append(complexity)
        chaos_scores.append(chaos)
        
        print(f"{name:<20} {quality:<18.4f} {complexity:<18.4f} {chaos:<18.4f}")
    
    print("-" * 80)
    print()
    
    # Create 3D scatter plot
    fig = go.Figure()
    
    # Color scheme
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    sizes = [10, 10, 10, 10, 12, 10]  # Lorenz slightly larger
    
    fig.add_trace(go.Scatter3d(
        x=quality_scores,
        y=complexity_scores,
        z=chaos_scores,
        mode='markers+text',
        marker=dict(
            size=sizes,
            color=colors,
            opacity=0.8,
            line=dict(color='white', width=2),
        ),
        text=names,
        textposition='top center',
        textfont=dict(size=12, color='black'),
        hovertemplate='<b>%{text}</b><br>' +
                      'Data Quality: %{x:.4f}<br>' +
                      'Complexity: %{y:.4f}<br>' +
                      'Chaos: %{z:.4f}<extra></extra>',
        showlegend=False
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='<b>3D Metric Space: Series Characterization</b><br>' +
                 '<sub>Data Quality vs Complexity vs Chaotic Behavior</sub>',
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis=dict(
                title='Data Quality Score',
                titlefont=dict(size=12),
                backgroundcolor='rgba(230, 230,230, 0.5)',
                gridcolor='white',
                showbackground=True,
                zeroline=True,
            ),
            yaxis=dict(
                title='Complexity Score',
                titlefont=dict(size=12),
                backgroundcolor='rgba(230, 230, 230, 0.5)',
                gridcolor='white',
                showbackground=True,
                zeroline=True,
            ),
            zaxis=dict(
                title='Chaotic Behavior (λ_max)',
                titlefont=dict(size=12),
                backgroundcolor='rgba(230, 230, 230, 0.5)',
                gridcolor='white',
                showbackground=True,
                zeroline=True,
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3),
                center=dict(x=0, y=0, z=0),
            ),
            aspectmode='cube',
        ),
        width=1200,
        height=800,
        hovermode='closest',
        font=dict(size=11),
        margin=dict(l=0, r=0, b=0, t=100),
    )
    
    # Save visualization
    output_path = 'output/metric_space_3d.html'
    fig.write_html(output_path)
    print(f"✓ 3D metric space visualization saved to: {output_path}")
    print()
    
    # Print interpretation
    print("Interpretation of the 3D metric space:")
    print("-" * 80)
    print("• Constant series: Low complexity, zero chaos, lower quality (repetitive)")
    print("• Sine wave: Low complexity, low chaos (predictable, periodic)")
    print("• Multi-Frequency: Higher complexity, low chaos (multiple frequencies)")
    print("• Noisy Multi-Freq: Higher complexity with noise (randomness added)")
    print("• Lorenz: High chaos, medium complexity (chaotic attractor)")
    print("• White Noise: High complexity, highest chaos (maximum randomness)")
    print()


if __name__ == '__main__':
    main()
