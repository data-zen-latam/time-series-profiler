"""Visualization module.

Render 3D and 2D scatterplots using Plotly.
"""

import plotly.graph_objects as go
import numpy as np


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
