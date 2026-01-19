"""
Plot 3D metric space for M5 dataset with a radar panel.

Left: 3D scatter (Quality vs Complexity vs Chaos) for levels L0, L1, L2 and a sample of 10 L3 items.
Right: Radar chart showing mean metrics by level (L0-L3).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.complexity import spectral_entropy
from src.chaos import largest_lyapunov_exponent
from src.data_quality import aggregate_quality_score
from src.visualization import plot_3d_with_radar


def aggregate_m5(df, level, day_cols):
    if level == 'L0':
        total = df[day_cols].sum(axis=0).values
        return {'total': total}
    if level == 'L1':
        grouped = df.groupby('cat_id')[day_cols].sum()
        return {f"cat_{cat}": grouped.loc[cat].values for cat in grouped.index}
    if level == 'L2':
        grouped = df.groupby('dept_id')[day_cols].sum()
        return {f"dept_{dept}": grouped.loc[dept].values for dept in grouped.index}
    if level == 'L3':
        grouped = df.groupby('item_id')[day_cols].sum()
        # Sample 10 items for plotting
        items = np.random.choice(grouped.index, size=min(10, len(grouped)), replace=False)
        return {f"item_{item}": grouped.loc[item].values for item in items}
    raise ValueError(level)


def compute_metrics(series_dict):
    names, q, c, h = [], [], [], []
    for name, series in series_dict.items():
        # Basic guards
        if len(series) < 100:
            continue
        names.append(name)
        q.append(aggregate_quality_score(series))
        c.append(spectral_entropy(series))
        h.append(largest_lyapunov_exponent(series))
    return names, q, c, h


def main():
    np.random.seed(42)
    data_path = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'sales_train_validation.csv'
    df = pd.read_csv(data_path)
    day_cols = [c for c in df.columns if c.startswith('d_')]

    levels = ['L0', 'L1', 'L2', 'L3']
    all_names, all_q, all_c, all_h = [], [], [], []

    # Collect metrics across levels
    level_means = {}
    for lvl in levels:
        series_dict = aggregate_m5(df, lvl, day_cols)
        names, q, c, h = compute_metrics(series_dict)
        all_names += [f"{lvl}:{n}" for n in names]
        all_q += q
        all_c += c
        all_h += h
        if len(q) > 0:
            level_means[lvl] = {
                'Quality': float(np.nanmean(q)),
                'Complexity': float(np.nanmean(c)),
                'Chaos': float(np.nanmean(h))
            }

    # Radar metric sets by level
    radar_labels = list(level_means.keys())
    radar_sets = [level_means[k] for k in radar_labels]

    # Plot combined figure
    output = Path(__file__).parent / 'output' / 'm5_3d_with_radar.html'
    output.parent.mkdir(parents=True, exist_ok=True)

    plot_3d_with_radar(
        all_q, all_c, all_h,
        series_ids=all_names,
        radar_metric_sets=radar_sets,
        radar_labels=radar_labels,
        title='<b>M5 Metric Space</b><br><sub>Quality vs Complexity vs Chaos</sub>',
        output_path=str(output)
    )

    print(f"✓ M5 3D + radar saved to: {output}")


if __name__ == '__main__':
    main()
