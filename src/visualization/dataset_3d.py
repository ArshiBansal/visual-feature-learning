"""3D visualizations for CIFAR-10 data exploration."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go


def plot_rgb_distribution_3d(image: np.ndarray) -> go.Figure:
    """Place an image's pixels in RGB color space with visible spatial detail."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3 or any(size <= 0 for size in image.shape):
        raise ValueError("image must be a non-empty RGB image with shape (height, width, 3)")
    if not np.all(np.isfinite(image)):
        raise ValueError("image must contain finite pixel values")
    if image.dtype == np.uint8:
        rgb = image.astype(np.float32) / 255
    else:
        rgb = image.astype(np.float32)
        if np.any((rgb < 0) | (rgb > 1)):
            raise ValueError("floating-point pixel values must lie in [0, 1]")

    height, width, _ = rgb.shape
    rows, columns = np.indices((height, width))
    colors = np.rint(rgb.reshape(-1, 3) * 255).astype(np.uint8)
    color_strings = [f"rgb({red},{green},{blue})" for red, green, blue in colors]
    figure = go.Figure(
        go.Scatter3d(
            x=rgb[..., 0].ravel(),
            y=rgb[..., 1].ravel(),
            z=rgb[..., 2].ravel(),
            mode="markers",
            customdata=np.column_stack((rows.ravel(), columns.ravel())),
            marker={
                "size": 4,
                "color": color_strings,
                "opacity": 0.8,
            },
            hovertemplate=(
                "R: %{x:.2f}<br>G: %{y:.2f}<br>B: %{z:.2f}"
                "<br>Pixel row: %{customdata[0]}"
                "<br>Pixel column: %{customdata[1]}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title="Pixels in RGB color space",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=580,
        margin={"l": 0, "r": 0, "t": 55, "b": 0},
        scene={
            "xaxis_title": "Red intensity",
            "yaxis_title": "Green intensity",
            "zaxis_title": "Blue intensity",
            "xaxis": {"range": [0, 1]},
            "yaxis": {"range": [0, 1]},
            "zaxis": {"range": [0, 1]},
            "aspectmode": "cube",
        },
    )
    return figure


def plot_class_color_profiles_3d(
    images: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
) -> go.Figure:
    """Compare average RGB signatures in 3D; use point size for class support."""
    images = np.asarray(images)
    labels = np.asarray(labels)
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("images must have shape (samples, height, width, 3)")
    if labels.ndim != 1 or len(labels) != len(images):
        raise ValueError("labels must be one-dimensional and match images")
    if not np.all(np.isfinite(images)):
        raise ValueError("images must contain finite pixel values")

    rgb = images.astype(np.float32)
    if images.dtype == np.uint8:
        rgb /= 255
    elif np.any((rgb < 0) | (rgb > 1)):
        raise ValueError("floating-point pixel values must lie in [0, 1]")

    counts = np.asarray([(labels == index).sum() for index in range(len(class_names))])
    mean_rgb = np.zeros((len(class_names), 3), dtype=np.float64)
    for class_index in range(len(class_names)):
        if counts[class_index]:
            mean_rgb[class_index] = rgb[labels == class_index].mean(axis=(0, 1, 2))
    color_strings = [
        f"rgb({int(red * 255)},{int(green * 255)},{int(blue * 255)})"
        for red, green, blue in mean_rgb
    ]
    maximum_count = int(counts.max()) if counts.size else 0
    marker_size = (
        8 + 12 * np.sqrt(counts / maximum_count) if maximum_count else np.full(len(class_names), 8)
    )
    figure = go.Figure(
        go.Scatter3d(
            x=mean_rgb[:, 0],
            y=mean_rgb[:, 1],
            z=mean_rgb[:, 2],
            text=list(class_names),
            mode="markers+text",
            textposition="top center",
            customdata=np.column_stack((mean_rgb[:, 2], counts)),
            marker={
                "size": marker_size,
                "color": color_strings,
                "line": {"color": "white", "width": 1},
            },
            hovertemplate=(
                "%{text}<br>Mean red: %{x:.3f}<br>Mean green: %{y:.3f}"
                "<br>Mean blue: %{z:.3f}<br>Atlas samples: %{customdata[1]:,}"
                "<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title="Class color signatures and test-set coverage",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=580,
        margin={"l": 0, "r": 0, "t": 55, "b": 0},
        scene={
            "xaxis_title": "Mean red intensity",
            "yaxis_title": "Mean green intensity",
            "zaxis_title": "Mean blue intensity",
            "xaxis": {"range": [0, 1]},
            "yaxis": {"range": [0, 1]},
            "zaxis": {"range": [0, 1]},
            "aspectmode": "cube",
        },
    )
    return figure