"""Visualization module.

Render 3D and 2D scatterplots using Plotly.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Okabe-Ito colorblind-friendly palette
OKABE_ITO_COLORS = [
    '#0173B2',  # Dark blue
    '#029E73',  # Teal
    '#DE8F05',  # Orange
    '#CC78BC',  # Purple
    '#CA9161',  # Brown
    '#56B4E9',  # Light blue
    '#ECE133',  # Yellow
    '#949494',  # Grey
]


def _hex_to_rgba(hex_color, alpha=1.0):
    """Convert hex color like '#RRGGBB' to an rgba string with given alpha."""
    try:
        h = hex_color.lstrip('#')
        r = int(h[0:2], 16)
        g = int(h[2:4], 16)
        b = int(h[4:6], 16)
        return f'rgba({r}, {g}, {b}, {alpha})'
    except Exception:
        # Fallback to grey if parsing fails
        return f'rgba(148, 148, 148, {alpha})'


def plot_3d_scatter(quality_values, complexity_values, chaos_values, 
                    title, series_ids=None, output_path=None, 
                    colors=None, sizes=None, opacities=None,
                    camera_eye=None):
    """
    Create an interactive 3D scatterplot with colorblind-friendly styling.
    
    Args:
        quality_values: array-like, data quality scores (x-axis)
        complexity_values: array-like, complexity scores (y-axis)
        chaos_values: array-like, chaos scores (z-axis)
        title: str, plot title
        series_ids: array-like, optional series identifiers for legend
        output_path: str, optional file path to save HTML output
        colors: list, optional color palette (defaults to Okabe-Ito)
        sizes: list, optional marker sizes (defaults to varying 8-14)
        opacities: list, optional marker opacities (defaults to varying 0.6-1.0)
        camera_eye: dict, optional camera position (default: lower-left perspective)
        
    Returns:
        plotly.graph_objects.Figure: the figure object
    """
    quality_values = np.asarray(quality_values)
    complexity_values = np.asarray(complexity_values)
    chaos_values = np.asarray(chaos_values)
    n = len(quality_values)
    
    if series_ids is None:
        series_ids = [f"Series {i+1}" for i in range(n)]
    
    # Default styling
    if colors is None:
        colors = [OKABE_ITO_COLORS[i % len(OKABE_ITO_COLORS)] for i in range(n)]
    if sizes is None:
        sizes = [8 + (i % 7) for i in range(n)]  # Cycle 8-14
    if opacities is None:
        opacities = [0.6 + (i % 5) * 0.1 for i in range(n)]  # Cycle 0.6-1.0
    
    if camera_eye is None:
        camera_eye = dict(x=-1.5, y=-1.5, z=1.5)  # Lower-left perspective
    
    fig = go.Figure()
    
    # Add individual traces for legend support
    for i in range(n):
        fig.add_trace(go.Scatter3d(
            x=[quality_values[i]],
            y=[complexity_values[i]],
            z=[chaos_values[i]],
            mode='markers',
            marker=dict(
                size=sizes[i],
                color=colors[i],
                opacity=opacities[i],
                line=dict(color='white', width=2),
            ),
            name=series_ids[i],
            hovertemplate='<b>%{fullData.name}</b><br>' +
                          'Data Quality: %{x:.4f}<br>' +
                          'Complexity: %{y:.4f}<br>' +
                          'Chaos: %{z:.4f}<extra></extra>',
            showlegend=True
        ))
    
    # Layout with visible grid and legend
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        scene=dict(
            xaxis=dict(
                title='Data Quality Score',
                title_font=dict(size=12),
                backgroundcolor='rgba(230, 230, 230, 0.5)',
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
            camera=dict(eye=camera_eye, center=dict(x=0, y=0, z=0)),
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
    
    if output_path:
        fig.write_html(output_path)
    
    return fig


def plot_2d_projection(quality_values, complexity_values, 
                       series_ids=None, output_path=None, colors=None):
    """
    Create a 2D scatterplot of Data Quality vs Complexity.
    
    Fallback visualization when chaotic behavior scores are unavailable.
    
    Args:
        quality_values: array-like, data quality scores (x-axis)
        complexity_values: array-like, complexity scores (y-axis)
        series_ids: array-like, optional series identifiers
        output_path: str, optional file path to save output
        colors: list, optional color palette (defaults to Okabe-Ito)
        
    Returns:
        plotly.graph_objects.Figure: the figure object
    """
    quality_values = np.asarray(quality_values)
    complexity_values = np.asarray(complexity_values)
    n = len(quality_values)
    
    if series_ids is None:
        series_ids = [f"Series {i+1}" for i in range(n)]
    
    if colors is None:
        colors = [OKABE_ITO_COLORS[i % len(OKABE_ITO_COLORS)] for i in range(n)]
    
    fig = go.Figure()
    
    # Add individual traces for legend
    for i in range(n):
        fig.add_trace(go.Scatter(
            x=[quality_values[i]],
            y=[complexity_values[i]],
            mode='markers',
            marker=dict(
                size=12,
                color=colors[i],
                opacity=0.8,
                line=dict(color='white', width=2),
            ),
            name=series_ids[i],
            hovertemplate='<b>%{fullData.name}</b><br>' +
                          'Data Quality: %{x:.4f}<br>' +
                          'Complexity: %{y:.4f}<extra></extra>',
            showlegend=True
        ))
    
    fig.update_layout(
        title='<b>Data Quality vs Complexity</b>',
        xaxis=dict(title='Data Quality Score', gridcolor='rgba(200, 200, 200, 0.5)'),
        yaxis=dict(title='Complexity Score', gridcolor='rgba(200, 200, 200, 0.5)'),
        width=900,
        height=700,
        hovermode='closest',
        font=dict(size=11),
        showlegend=True,
        legend=dict(
            x=1.02,
            y=1,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1,
        ),
    )
    
    if output_path:
        fig.write_html(output_path)
    
    return fig

def plot_3d_with_radar(quality_values, complexity_values, chaos_values,
                       series_ids=None, radar_metric_sets=None, radar_labels=None,
                       title='', output_path=None, colors=None):
    """
    Create a combined figure with 3D scatter plot (left) and radar chart (right).
    
    Args:
        quality_values: array-like, data quality scores (x-axis)
        complexity_values: array-like, complexity scores (y-axis)
        chaos_values: array-like, chaos scores (z-axis)
        series_ids: array-like, optional series identifiers for legend
        radar_metric_sets: list of dicts, each dict has metric names as keys and values
        radar_labels: list of str, labels for radar traces (e.g., level names)
        title: str, plot title
        output_path: str, optional file path to save HTML output
        colors: list, optional color palette (defaults to Okabe-Ito)
        
    Returns:
        plotly.graph_objects.Figure: the figure object with subplots
    """
    quality_values = np.asarray(quality_values)
    complexity_values = np.asarray(complexity_values)
    chaos_values = np.asarray(chaos_values)
    n = len(quality_values)
    
    if series_ids is None:
        series_ids = [f"Series {i+1}" for i in range(n)]
    
    # Default styling for 3D scatter
    if colors is None:
        colors = [OKABE_ITO_COLORS[i % len(OKABE_ITO_COLORS)] for i in range(n)]
    sizes = [8 + (i % 7) for i in range(n)]
    opacities = [0.6 + (i % 5) * 0.1 for i in range(n)]
    
    # If saving to HTML, build two figures side-by-side with independent legends
    if output_path:
        fig3d = go.Figure()
        for i in range(n):
            fig3d.add_trace(go.Scatter3d(
                x=[quality_values[i]],
                y=[complexity_values[i]],
                z=[chaos_values[i]],
                mode='markers',
                marker=dict(
                    size=sizes[i],
                    color=colors[i],
                    opacity=opacities[i],
                    line=dict(color='white', width=2),
                ),
                name=series_ids[i],
                hovertemplate='<b>%{fullData.name}</b><br>' +
                              'Data Quality: %{x:.4f}<br>' +
                              'Complexity: %{y:.4f}<br>' +
                              'Chaos: %{z:.4f}<extra></extra>',
                showlegend=True
            ))

        fig3d.update_layout(
            title=dict(text=title, x=0.5, xanchor='center'),
            scene=dict(
                xaxis=dict(
                    title='Data Quality Score',
                    title_font=dict(size=12),
                    backgroundcolor='rgba(230, 230, 230, 0.5)',
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
                camera=dict(eye=dict(x=-1.5, y=-1.5, z=1.5), center=dict(x=0, y=0, z=0)),
                aspectmode='cube',
            ),
            width=900,
            height=700,
            hovermode='closest',
            font=dict(size=11),
            margin=dict(l=50, r=50, b=50, t=80),
            showlegend=True,
        )

        figPolar = go.Figure()
        if radar_metric_sets and radar_labels:
            metric_names = list(radar_metric_sets[0].keys()) if radar_metric_sets else []
            for idx, (metric_set, label) in enumerate(zip(radar_metric_sets, radar_labels)):
                r_values = [metric_set.get(m, 0) for m in metric_names]
                line_color = colors[idx % len(colors)] if colors is not None else OKABE_ITO_COLORS[idx % len(OKABE_ITO_COLORS)]
                initial_visible = True if str(label).strip().lower() == 'white noise' else 'legendonly'
                figPolar.add_trace(go.Scatterpolar(
                    r=r_values,
                    theta=metric_names,
                    mode='lines',
                    fill='toself',
                    fillcolor=_hex_to_rgba(line_color, alpha=0.15),
                    line=dict(color=line_color, width=2.5, dash='solid'),
                    name=label,
                    opacity=1.0,
                    hovertemplate='<b>%{fullData.name}</b><br>' + '%{theta}: %{r:.4f}<extra></extra>',
                    showlegend=True,
                    visible=initial_visible,
                ))

        figPolar.update_layout(
            title=dict(text='Radar', x=0.5, xanchor='center'),
            width=900,
            height=700,
            hovermode='closest',
            font=dict(size=11),
            margin=dict(l=50, r=50, b=50, t=80),
            showlegend=True,
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickfont=dict(size=10),
                ),
            ),
        )

        # Build combined HTML with two independent legends (no hover script)
        html_left = fig3d.to_html(include_plotlyjs='cdn', full_html=False, div_id='plotly-3d')
        html_right = figPolar.to_html(include_plotlyjs=False, full_html=False, div_id='plotly-radar')
        full_html = (
            "<!DOCTYPE html><html><head><meta charset='utf-8'><title>3D + Radar</title>"+
            "<style>body{margin:0;font-family:sans-serif;} .row{display:flex;flex-direction:row;} .col{flex:1;padding:10px;} </style></head><body>"+
            "<div class='row'>"+
            "<div class='col'>" + html_left + "</div>"+
            "<div class='col'>" + html_right + "</div>"+
            "</div>"+
            "</body></html>"
        )
        with open(output_path, 'w') as f:
            f.write(full_html)
        return fig3d

    # Fallback for environments without output_path (e.g., inline notebooks):
    # Create subplots: 1 row, 2 cols (3d + regular)
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatterpolar'}]],
        column_widths=[0.5, 0.5],
        horizontal_spacing=0.12
    )
    
    # Add 3D scatter traces
    for i in range(n):
        fig.add_trace(go.Scatter3d(
            x=[quality_values[i]],
            y=[complexity_values[i]],
            z=[chaos_values[i]],
            mode='markers',
            marker=dict(
                size=sizes[i],
                color=colors[i],
                opacity=opacities[i],
                line=dict(color='white', width=2),
            ),
            name=series_ids[i],
            hovertemplate='<b>%{fullData.name}</b><br>' +
                          'Data Quality: %{x:.4f}<br>' +
                          'Complexity: %{y:.4f}<br>' +
                          'Chaos: %{z:.4f}<extra></extra>',
            showlegend=True,
            # Group all 3D points under one legend section
            legendgroup='3D',
            legendgrouptitle_text='3D Scatter' if i == 0 else None,
        ), row=1, col=1)
    
    # Add radar chart traces if provided
    if radar_metric_sets and radar_labels:
        # Extract metric names from first set
        metric_names = list(radar_metric_sets[0].keys()) if radar_metric_sets else []
        
        for idx, (metric_set, label) in enumerate(zip(radar_metric_sets, radar_labels)):
            r_values = [metric_set.get(m, 0) for m in metric_names]

            # Use same color as corresponding 3D scatter when labels align; else fallback palette
            if series_ids is not None and len(series_ids) == len(radar_labels) and colors is not None:
                line_color = colors[idx]
            else:
                line_color = OKABE_ITO_COLORS[idx % len(OKABE_ITO_COLORS)]
            
            fig.add_trace(go.Scatterpolar(
                r=r_values,
                theta=metric_names,
                mode='lines',
                fill='toself',
                fillcolor=_hex_to_rgba(line_color, alpha=0.15),  # low opacity by default
                line=dict(color=line_color, width=2.5, dash='solid'),  # solid border
                name=label,
                opacity=1.0,
                hovertemplate='<b>%{fullData.name}</b><br>' +
                              '%{theta}: %{r:.4f}<extra></extra>',
                showlegend=True,
                # Group all radar traces under one legend section
                legendgroup='Radar',
                legendgrouptitle_text='Radar' if idx == 0 else None,
                visible=True if str(label).strip().lower() == 'white noise' else 'legendonly',
            ), row=1, col=2)
    
    # Update 3D scene layout
    fig.update_scenes(
        xaxis=dict(
            title='Data Quality Score',
            title_font=dict(size=12),
            backgroundcolor='rgba(230, 230, 230, 0.5)',
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
        camera=dict(eye=dict(x=-1.5, y=-1.5, z=1.5), center=dict(x=0, y=0, z=0)),
        aspectmode='cube',
        row=1, col=1
    )
    
    # Update polar (radar) layout
    fig.update_polars(
        radialaxis=dict(
            visible=True,
            range=[0, 1],
            tickfont=dict(size=10),
        ),
        row=1, col=2
    )
    
    # Overall layout
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        width=1800,
        height=700,
        hovermode='closest',
        font=dict(size=11),
        margin=dict(l=50, r=50, b=50, t=100),
        showlegend=True,
        legend=dict(
            x=1.05,
            y=1,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1,
            font=dict(size=10),
            tracegroupgap=10,
        ),
    )
    
    if output_path:
        fig.write_html(output_path)
    
    return fig