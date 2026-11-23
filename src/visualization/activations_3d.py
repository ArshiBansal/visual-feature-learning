"""Interactive Plotly visualizations for convolutional activations."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def plot_activation_surface(
    activation: np.ndarray,
    *,
    title: str = "3D activation surface",
    colorscale: str = "Viridis",
) -> go.Figure:
    """Create a 3D surface figure from one (height, width) feature map."""
    activation = np.asarray(activation, dtype=np.float64)
    if activation.ndim != 2 or any(size <= 0 for size in activation.shape):
        raise ValueError("activation must be a non-empty (height, width) array")
    if not np.all(np.isfinite(activation)):
        raise ValueError("activation must contain only finite values")

    height, width = activation.shape
    fig = go.Figure(
        data=[
            go.Surface(
                x=np.arange(width),
                y=np.arange(height),
                z=activation,
                colorscale=colorscale,
                colorbar={"title": "Activation"},
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene={
            "xaxis_title": "Pixel x",
            "yaxis_title": "Pixel y",
            "zaxis_title": "Activation",
        },
        height=650,
    )
    return fig


def plot_activation_volume(
    volume: np.ndarray,
    *,
    title: str = "3D pooled activation volume",
    colorscale: str = "Viridis",
    opacity: float = 0.12,
    surface_count: int = 12,
) -> go.Figure:
    """Create a Plotly volume from an (height, width, channels) activation array."""
    volume = np.asarray(volume, dtype=np.float64)
    if volume.ndim != 3 or any(size <= 0 for size in volume.shape):
        raise ValueError("volume must be a non-empty (height, width, channels) array")
    if not np.all(np.isfinite(volume)):
        raise ValueError("volume must contain only finite values")
    if not 0 <= opacity <= 1:
        raise ValueError("opacity must be between 0 and 1")
    if surface_count <= 0:
        raise ValueError("surface_count must be positive")

    positive_values = volume[volume > 0]
    if positive_values.size == 0:
        raise ValueError("volume must contain at least one positive activation")

    height, width, channels = volume.shape
    grid_y, grid_x, grid_channel = np.mgrid[
        0:height,
        0:width,
        0:channels,
    ]
    isomin = float(np.percentile(positive_values, 50))
    isomax = float(positive_values.max())
    if isomin >= isomax:
        isomin = isomax * 0.5

    fig = go.Figure(
        data=[
            go.Volume(
                x=grid_x.ravel(),
                y=grid_y.ravel(),
                z=grid_channel.ravel(),
                value=volume.ravel(),
                isomin=isomin,
                isomax=isomax,
                opacity=opacity,
                surface_count=surface_count,
                colorscale=colorscale,
                colorbar={"title": "Activation"},
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene={
            "xaxis_title": "Pixel x",
            "yaxis_title": "Pixel y",
            "zaxis_title": "Feature channel",
        },
        height=700,
    )
    return fig