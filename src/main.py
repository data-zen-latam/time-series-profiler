"""Entry point and pipeline orchestration for the Time Series Profiler."""

import argparse
import logging
import sys
import numpy as np
from pathlib import Path

from src.utils import load_csv
from src.complexity import spectral_entropy, dominant_frequency_ratio
from src.chaos import largest_lyapunov_exponent
from src.data_quality import aggregate_quality_score
from src.visualization import plot_3d_scatter, plot_2d_projection

logger = logging.getLogger(__name__), plot_2d_projection


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Classify time series by complexity, chaos, and data quality."
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input CSV file"
    )
    parser.add_argument(
        "--output", type=str, default="metrics.csv", help="Output path for metrics CSV"
    )
    parser.add_argument(
        "--plot3d", type=str, default="scatter3d.html", help="Output path for 3D scatterplot"
    )
    parser.add_argument(
        "--window-size", type=int, default=256, help="Window size for sliding analysis"
    )
    parser.add_argument(
        "--embedding-dim", type=int, default=5, help="Embedding dimension for chaos metrics"
    )
    parser.add_argument(
        "--delay", type=int, default=1, help="Delay for delay embedding"
    )
    parser.add_argument(
        "--projection", type=str, choices=["xy", "xz", "yz"],
        help="Optional 2D projection to display"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Show interactive plots"
    )
    
    args = parser.parse_args()
    
    # Load data
    try:
        df = load_csv(args.input)
    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading CSV: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Compute metrics for a single series
    print("Computing metrics...")
    try:
        values = df["value"].values

        complexity = spectral_entropy(values)
        chaos = largest_lyapunov_exponent(values, args.embedding_dim, args.delay)
        quality = aggregate_quality_score(values)

        results = [{
            "complexity": complexity,
            "chaos": chaos,
            "quality": quality,
        }]
    except Exception as e:
        print(f"Error computing metrics: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Export metrics
    try:
        import pandas as pd
        results_df = pd.DataFrame(results)
        results_df.to_csv(args.output, index=False)
        print(f"Metrics saved to {args.output}")
    except Exception as e:
        print(f"Error exporting metrics: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Create 3D plot or 2D projection if chaos unavailable
    try:
        # Check if chaos values are valid (not all NaN)
        has_valid_chaos = np.isfinite(results_df["chaos"].values).any()
        
        if has_valid_chaos:
            plot_3d_scatter(
                complexity_values=results_df["complexity"].values,
                quality_values=results_df["quality"].values,
                chaos_values=results_df["chaos"].values,
                series_ids=None,
                output_path=args.plot3d,
            )
            print(f"3D scatterplot saved to {args.plot3d}")
        else:
            logger.warning("Chaos metrics unavailable; creating 2D projection (complexity vs quality)")
            plot_2d_projection(
                x_values=results_df["complexity"].values,
                y_values=results_df["quality"].values,
                series_ids=None,
                output_path=args.plot3d,
                x_label="Complexity",
                y_label="Quality",
            )
            print(f"2D scatterplot (chaos unavailable) saved to {args.plot3d}")
    except Exception as e:
        print(f"Warning: Could not create plot: {e}", file=sys.stderr)
    
    print("Done!")


if __name__ == "__main__":
    main()
