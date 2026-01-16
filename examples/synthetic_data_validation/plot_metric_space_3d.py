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
from src.visualization import plot_3d_scatter


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
    
    # Create 3D scatter plot using visualization module
    output_file = Path(__file__).parent / 'output' / 'metric_space_3d.html'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    plot_3d_scatter(
        quality_scores,
        complexity_scores,
        chaos_scores,
        title='<b>3D Metric Space: Series Characterization</b><br>' +
              '<sub>Data Quality vs Complexity vs Chaotic Behavior</sub>',
        series_ids=names,
        output_path=str(output_file)
    )
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
