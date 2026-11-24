"""3D views of learned filters and channel connections."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def plot_filter_kernel_3d(
    kernel: np.ndarray,
    *,
    title: str,
) -> go.Figure:
    """Render a convolution kernel as a color-coded 3D lattice."""
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.ndim != 3 or any(size <= 0 for size in kernel.shape):
        raise ValueError("kernel must have shape (channels, height, width)")
    if not np.all(np.isfinite(kernel)):
        raise ValueError("kernel must contain only finite values")

    channel, row, column = np.indices(kernel.shape)
    values = kernel.ravel()
    magnitude = float(np.max(np.abs(values)))
    marker_size = 6 + 10 * np.abs(values) / magnitude if magnitude else 6

    figure = go.Figure(
        go.Scatter3d(
            x=column.ravel(),
            y=row.ravel(),
            z=channel.ravel(),
            mode="markers",
            customdata=np.column_stack(
                (channel.ravel(), row.ravel(), column.ravel())
            ),
            marker={
                "size": marker_size,
                "color": values,
                "colorscale": "RdBu",
                "cmin": -magnitude if magnitude else -1,
                "cmax": magnitude if magnitude else 1,
                "colorbar": {"title": "Weight"},
                "line": {
                    "color": "rgba(255,255,255,0.65)",
                    "width": 1,
                },
            },
            hovertemplate=(
                "Input channel: %{customdata[0]}"
                "<br>Kernel row: %{customdata[1]}"
                "<br>Kernel column: %{customdata[2]}"
                "<br>Weight: %{marker.color:.4f}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=540,
        margin={"l": 0, "r": 0, "t": 55, "b": 0},
        scene={
            "xaxis_title": "Kernel column",
            "yaxis_title": "Kernel row",
            "zaxis_title": "Input channel",
            "aspectmode": "auto",
        },
    )
    return figure


def plot_filter_channel_strength_3d(weights: np.ndarray) -> go.Figure:
    """Show RMS filter weights over input/output channel indices."""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 4 or any(size <= 0 for size in weights.shape):
        raise ValueError(
            "weights must have shape "
            "(out_channels, in_channels, height, width)"
        )
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights must contain only finite values")

    strength = np.sqrt(np.mean(np.square(weights), axis=(2, 3)))
    output_filter, input_channel = np.indices(strength.shape)
    maximum = float(strength.max())
    marker_size = (
        5 + 10 * strength.ravel() / maximum
        if maximum
        else 5
    )

    figure = go.Figure(
        go.Scatter3d(
            x=input_channel.ravel(),
            y=output_filter.ravel(),
            z=strength.ravel(),
            mode="markers",
            marker={
                "size": marker_size,
                "color": strength.ravel(),
                "colorscale": "Viridis",
                "colorbar": {"title": "RMS weight"},
            },
            hovertemplate=(
                "Input channel: %{x}"
                "<br>Output filter: %{y}"
                "<br>RMS weight: %{z:.5f}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title="Second-layer filter strength",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=620,
        margin={"l": 0, "r": 0, "t": 55, "b": 0},
        scene={
            "xaxis_title": "Input feature-map channel",
            "yaxis_title": "Output filter",
            "zaxis_title": "RMS weight",
            "aspectmode": "auto",
        },
    )
    return figure