"""3D views of classification performance."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go


def _validate_confusion(
    confusion: np.ndarray,
    class_names: Sequence[str],
) -> np.ndarray:
    confusion = np.asarray(confusion, dtype=np.float64)
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1] or confusion.shape[0] == 0:
        raise ValueError("confusion must be a non-empty square matrix")
    if len(class_names) != confusion.shape[0]:
        raise ValueError("class_names must match the confusion matrix size")
    if not np.all(np.isfinite(confusion)) or np.any(confusion < 0):
        raise ValueError("confusion must contain finite, non-negative counts")
    return confusion


def plot_confusion_matrix_3d(
    confusion: np.ndarray,
    class_names: Sequence[str],
) -> go.Figure:
    """Render actual/predicted class pairs as a 3D confusion point cloud."""
    confusion = _validate_confusion(confusion, class_names)
    actual, predicted = np.indices(confusion.shape)
    counts = confusion.ravel()
    maximum = float(counts.max())
    relative = counts / maximum if maximum else np.zeros_like(counts)
    labels = np.asarray(class_names, dtype=object)
    figure = go.Figure(
        go.Scatter3d(
            x=predicted.ravel(),
            y=actual.ravel(),
            z=counts,
            mode="markers",
            customdata=np.column_stack((labels[predicted.ravel()], labels[actual.ravel()])),
            marker={
                "size": 4 + 13 * np.sqrt(relative),
                "color": counts,
                "colorscale": "Turbo",
                "colorbar": {"title": "Samples"},
            },
            hovertemplate=(
                "Predicted: %{customdata[0]}<br>Actual: %{customdata[1]}"
                "<br>Samples: %{z:,}<extra></extra>"
            ),
        )
    )
    ticks = list(range(len(class_names)))
    figure.update_layout(
        title="Confusion landscape by actual and predicted class",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=640,
        margin={"l": 0, "r": 0, "t": 55, "b": 0},
        scene={
            "xaxis_title": "Predicted class",
            "yaxis_title": "Actual class",
            "zaxis_title": "Sample count",
            "xaxis": {"tickmode": "array", "tickvals": ticks, "ticktext": list(class_names)},
            "yaxis": {"tickmode": "array", "tickvals": ticks, "ticktext": list(class_names)},
            "aspectmode": "auto",
        },
    )
    return figure


def plot_per_class_accuracy_3d(
    confusion: np.ndarray,
    class_names: Sequence[str],
) -> go.Figure:
    """Compare class recall against the number of test examples per class."""
    confusion = _validate_confusion(confusion, class_names)
    support = confusion.sum(axis=1)
    accuracy = np.divide(
        np.diag(confusion),
        support,
        out=np.zeros_like(support),
        where=support != 0,
    )
    class_index = np.arange(len(class_names))
    maximum = float(support.max())
    marker_size = 8 + 10 * np.sqrt(support / maximum) if maximum else 8
    figure = go.Figure(
        go.Scatter3d(
            x=class_index,
            y=accuracy * 100,
            z=support,
            text=list(class_names),
            mode="markers+text",
            textposition="top center",
            marker={
                "size": marker_size,
                "color": accuracy * 100,
                "cmin": 0,
                "cmax": 100,
                "colorscale": "Viridis",
                "colorbar": {"title": "Accuracy (%)"},
            },
            hovertemplate=("%{text}<br>Accuracy: %{y:.1f}%<br>Test samples: %{z:,}<extra></extra>"),
        )
    )
    figure.update_layout(
        title="Per-class accuracy and test-set support",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=620,
        margin={"l": 0, "r": 0, "t": 55, "b": 0},
        scene={
            "xaxis_title": "Class index",
            "yaxis_title": "Accuracy (%)",
            "zaxis_title": "Test samples",
            "xaxis": {
                "tickmode": "array",
                "tickvals": class_index.tolist(),
                "ticktext": list(class_names),
            },
            "aspectmode": "auto",
        },
    )
    return figure