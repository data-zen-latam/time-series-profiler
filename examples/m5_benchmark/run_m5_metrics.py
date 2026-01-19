"""
Compute chaos and complexity metrics on M5 dataset at different hierarchical levels.

Replicates the analysis from AWS paper "Time Series Forecastability Measures"
comparing spectral predictability and Lyapunov exponents across aggregation levels.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.complexity import spectral_entropy
from src.chaos import largest_lyapunov_exponent


def load_m5_data(file_path):
    """Load M5 sales data."""
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} series from {file_path}")
    return df


def aggregate_to_level(df, level):
    """
    Aggregate M5 data to hierarchical level.
    
    Levels:
    - L0: Total (all series summed)
    - L1: Category (cat_id)
    - L2: Department (dept_id)
    - L3: Product/Item (item_id)
    - L12: Store-Category (store_id + cat_id)
    """
    # Extract daily sales columns (d_1, d_2, ..., d_1913)
    day_cols = [col for col in df.columns if col.startswith('d_')]
    
    if level == 'L0':
        # Total: sum all series
        total = df[day_cols].sum(axis=0).values
        return {'total': total}
    
    elif level == 'L1':
        # Category level
        grouped = df.groupby('cat_id')[day_cols].sum()
        return {f"cat_{cat}": grouped.loc[cat].values for cat in grouped.index}
    
    elif level == 'L2':
        # Department level
        grouped = df.groupby('dept_id')[day_cols].sum()
        return {f"dept_{dept}": grouped.loc[dept].values for dept in grouped.index}
    
    elif level == 'L3':
        # Item level (sample subset due to size and computational cost)
        grouped = df.groupby('item_id')[day_cols].sum()
        # Sample 50 items for better statistics
        sampled_items = np.random.choice(grouped.index, size=min(50, len(grouped)), replace=False)
        return {f"item_{item}": grouped.loc[item].values for item in sampled_items}
    
    elif level == 'L12':
        # Store-Category level
        df['store_cat'] = df['store_id'] + '_' + df['cat_id']
        grouped = df.groupby('store_cat')[day_cols].sum()
        return {f"store_cat_{sc}": grouped.loc[sc].values for sc in grouped.index}
    
    else:
        raise ValueError(f"Unknown level: {level}")


def compute_metrics_for_level(series_dict, level_name):
    """Compute chaos and complexity metrics for all series at a level."""
    results = []
    
    for name, series in series_dict.items():
        # Skip series with insufficient data
        if len(series) < 100:
            continue
        
        # Compute metrics
        complexity = spectral_entropy(series)
        chaos = largest_lyapunov_exponent(series)
        
        # Only include valid results
        if not np.isnan(complexity) and not np.isnan(chaos):
            results.append({
                'level': level_name,
                'series': name,
                'complexity': complexity,
                'chaos': chaos
            })
    
    return results


def main():
    """Run M5 benchmark analysis."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load data
    data_path = Path(__file__).parent.parent.parent / "data" / "raw" / "sales_train_validation.csv"
    df = load_m5_data(data_path)
    
    # Define levels to analyze
    levels = ['L0', 'L1', 'L2', 'L3']
    
    all_results = []
    
    print("\n" + "="*80)
    print("Computing metrics for each hierarchical level...")
    print("="*80 + "\n")
    
    for level in levels:
        print(f"Processing {level}...")
        
        # Aggregate data to this level
        series_dict = aggregate_to_level(df, level)
        print(f"  {len(series_dict)} series at this level")
        
        # Compute metrics
        results = compute_metrics_for_level(series_dict, level)
        all_results.extend(results)
        
        print(f"  Computed metrics for {len(results)} series\n")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    print("\n" + "="*80)
    print("SUMMARY BY LEVEL")
    print("="*80 + "\n")
    
    summary_data = []
    print("Spectral Predictability (Complexity):")
    print("-" * 50)
    for level in levels:
        level_data = results_df[results_df['level'] == level]
        if len(level_data) > 0:
            comp_mean = level_data['complexity'].mean()
            comp_std = level_data['complexity'].std()
            chaos_mean = level_data['chaos'].mean()
            chaos_std = level_data['chaos'].std()
            n = len(level_data)
            
            summary_data.append({
                'Level': level,
                'N Series': n,
                'Complexity Mean': comp_mean,
                'Complexity Std': comp_std,
                'Chaos Mean': chaos_mean,
                'Chaos Std': chaos_std
            })
            
            print(f"{level} (total)     {comp_mean:.3f} ± {comp_std:.3f}")
    
    print("\n\nLyapunov Exponents (Chaos):")
    print("-" * 50)
    for level in levels:
        level_data = results_df[results_df['level'] == level]
        if len(level_data) > 0:
            chaos_mean = level_data['chaos'].mean()
            chaos_std = level_data['chaos'].std()
            print(f"{level} (total)     {chaos_mean:.3f} ± {chaos_std:.3f}")
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save detailed results
    output_path = Path(__file__).parent / "output" / "m5_metrics.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\n\nDetailed results saved to: {output_path}")
    
    # Save markdown summary
    md_path = output_path.parent / "m5_metrics.md"
    with open(md_path, 'w') as f:
        f.write("# M5 Dataset - Time Series Metrics Analysis\n\n")
        f.write("## Spectral Predictability (Complexity)\n\n")
        f.write("| Level | Mean ± Std |\n")
        f.write("|-------|------------|\n")
        for _, row in summary_df.iterrows():
            f.write(f"| {row['Level']} | {row['Complexity Mean']:.3f} ± {row['Complexity Std']:.3f} |\n")
        
        f.write("\n## Lyapunov Exponents (Chaos)\n\n")
        f.write("| Level | Mean ± Std |\n")
        f.write("|-------|------------|\n")
        for _, row in summary_df.iterrows():
            f.write(f"| {row['Level']} | {row['Chaos Mean']:.3f} ± {row['Chaos Std']:.3f} |\n")
    
    print(f"\n\nMarkdown summary saved to: {md_path}")


if __name__ == "__main__":
    main()
