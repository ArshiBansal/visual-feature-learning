"""3D visualizations for experiment comparisons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import plotly.graph_objects as go


def plot_ablation_tradeoffs_3d(
    results: Sequence[Mapping[str, object]],
) -> go.Figure:
    """Compare model capacity, accuracy, and training cost in 3D."""
    if not results:
        raise ValueError("results must contain at least one experiment")

    parameters = np.asarray(
        [row["parameter_count"] for row in results],
        dtype=np.float64,
    )
    accuracies = np.asarray(
        [row["test_accuracy"] for row in results],
        dtype=np.float64,
    )
    minutes = (
        np.asarray(
            [row["training_seconds"] for row in results],
            dtype=np.float64,
        )
        / 60
    )
    losses = np.asarray(
        [row["test_loss"] for row in results],
        dtype=np.float64,
    )

    if not all(
        np.all(np.isfinite(values))
        for values in (parameters, accuracies, minutes, losses)
    ):
        raise ValueError("ablation results must contain finite values")

    figure = go.Figure(
        go.Scatter3d(
            x=parameters,
            y=accuracies,
            z=minutes,
            text=[str(row["name"]) for row in results],
            mode="markers+text",
            textposition="top center",
            marker={
                "size": 8,
                "color": losses,
                "colorscale": "Plasma",
                "colorbar": {"title": "Test loss"},
                "line": {"color": "white", "width": 1},
            },
            hovertemplate=(
                "%{text}<br>Parameters: %{x:,}"
                "<br>Test accuracy: %{y:.2%}"
                "<br>Training time: %{z:.1f} min"
                "<br>Test loss: %{marker.color:.4f}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title="Ablation trade-offs: capacity, accuracy, and training time",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=640,
        margin={"l": 0, "r": 0, "t": 55, "b": 0},
        scene={
            "xaxis_title": "Parameters",
            "yaxis_title": "Test accuracy",
            "zaxis_title": "Training time (minutes)",
            "aspectmode": "auto",
        },
    )
    return figure