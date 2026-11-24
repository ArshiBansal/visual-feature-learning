"""3D visualizations of model training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import plotly.graph_objects as go


def plot_training_trajectory_3d(
    history: Sequence[Mapping[str, float]],
) -> go.Figure:
    """Plot each epoch in loss/accuracy space, connected in training order."""
    if not history:
        raise ValueError("history must contain at least one epoch")

    epochs = np.asarray(
        [row["epoch"] for row in history],
        dtype=np.float64,
    )
    losses = np.asarray(
        [row["train_loss"] for row in history],
        dtype=np.float64,
    )
    accuracies = np.asarray(
        [row["train_accuracy"] for row in history],
        dtype=np.float64,
    )

    if not all(
        np.all(np.isfinite(values))
        for values in (epochs, losses, accuracies)
    ):
        raise ValueError("training history values must be finite")

    figure = go.Figure(
        go.Scatter3d(
            x=epochs,
            y=losses,
            z=accuracies,
            mode="lines+markers",
            text=[f"Epoch {epoch:g}" for epoch in epochs],
            marker={
                "size": 6,
                "color": epochs,
                "colorscale": "Viridis",
                "colorbar": {"title": "Epoch"},
            },
            line={"color": "#38bdf8", "width": 5},
            hovertemplate=(
                "%{text}<br>Training loss: %{y:.4f}"
                "<br>Training accuracy: %{z:.2%}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title="Learning trajectory through loss and accuracy",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=620,
        margin={"l": 0, "r": 0, "t": 55, "b": 0},
        scene={
            "xaxis_title": "Epoch",
            "yaxis_title": "Training loss",
            "zaxis_title": "Training accuracy",
            "aspectmode": "auto",
        },
    )
    return figure