"""
Generate 3D scatter plot: Data Quality vs Complexity vs Chaotic Behavior.

Each point represents one synthetic time series, positioned according to:
- X-axis: Data Quality Score
- Y-axis: Complexity Score  
- Z-axis: Chaotic Behavior (Largest Lyapunov Exponent)
"""

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from generators import (
    generate_constant,
    generate_lorenz,
    generate_multi_frequency,
    generate_noisy_multi_frequency,
    generate_sine_wave,
    generate_white_noise,
)
from src.complexity import spectral_entropy
from src.chaos import largest_lyapunov_exponent
from src.data_quality import aggregate_quality_score
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
        complexity = spectral_entropy(series)
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
    
    # Okabe-Ito colorblind-friendly palette + size/opacity for redundancy
    # Designed for color blindness (protanopia, deuteranopia, tritanopia)
    colors = [
        '#0173B2',  # Dark blue     - Constant
        '#029E73',  # Teal          - Sine
        '#DE8F05',  # Orange        - Multi-Freq
        '#CC78BC',  # Purple        - Noisy Multi-Freq
        '#CA9161',  # Brown         - Lorenz
        '#56B4E9',  # Light blue    - White Noise
    ]
    sizes = [8, 10, 11, 12, 14, 9]  # Varying sizes for visual distinction
    opacities = [0.6, 0.7, 0.8, 0.9, 1.0, 0.8]  # Varying opacity for depth perception
    
    # Add individual traces for each series (enables legend)
    for i, name in enumerate(names):
        fig.add_trace(go.Scatter3d(
            x=[quality_scores[i]],
            y=[complexity_scores[i]],
            z=[chaos_scores[i]],
            mode='markers',
            marker=dict(
                size=sizes[i],
                color=colors[i],
                opacity=opacities[i],
                line=dict(color='white', width=2),
            ),
            name=name,
            hovertemplate='<b>%{fullData.name}</b><br>' +
                          'Data Quality: %{x:.4f}<br>' +
                          'Complexity: %{y:.4f}<br>' +
                          'Chaos: %{z:.4f}<extra></extra>',
            showlegend=True
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
                title_font=dict(size=12),
                backgroundcolor='rgba(230, 230,230, 0.5)',
                gridcolor='rgba(100, 100, 100, 0.9)',
                gridwidth=2.5,
                showbackground=True,
                zeroline=True,
            ),
            yaxis=dict(
                title='Complexity Score',
                title_font=dict(size=12),
                backgroundcolor='rgba(230, 230, 230, 0.5)',
                gridcolor='rgba(100, 100, 100, 0.9)',
                gridwidth=2.5,
                showbackground=True,
                zeroline=True,
            ),
            zaxis=dict(
                title='Chaotic Behavior (λ_max)',
                title_font=dict(size=12),
                backgroundcolor='rgba(230, 230, 230, 0.5)',
                gridcolor='rgba(100, 100, 100, 0.9)',
                gridwidth=2.5,
                showbackground=True,
                zeroline=True,
            ),
            camera=dict(
                eye=dict(x=-1.5, y=-1.5, z=1.5),
                center=dict(x=0, y=0, z=0),
            ),
            aspectmode='cube',
        ),
        width=1200,
        height=800,
        hovermode='closest',
        font=dict(size=11),
        margin=dict(l=0, r=100, b=0, t=100),
        showlegend=True,
        legend=dict(
            x=0.85,
            y=0.95,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1,
        ),
    )
    
    # Save visualization
    output_file = Path(__file__).parent / 'output' / 'metric_space_3d.html'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_file)
    print(f"✓ 3D metric space visualization saved to: {output_file.absolute()}")
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
