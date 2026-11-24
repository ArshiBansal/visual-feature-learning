"""3D views of the CIFAR-10 network architecture."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def plot_model_architecture_3d() -> go.Figure:
    """Illustrate configured tensor stages as connected 3D feature groups."""
    stages = [
        ("RGB input", "3 × 32 × 32", 3),
        ("Convolution", "16 × 32 × 32", 16),
        ("Max pool", "16 × 16 × 16", 16),
        ("Convolution", "32 × 16 × 16", 32),
        ("Max pool", "32 × 8 × 8", 32),
        ("Classifier", "10 logits", 10),
    ]
    representatives = [3, 5, 5, 6, 6, 4]
    points: list[tuple[float, float, float, str, str]] = []

    for stage_index, ((name, shape, channels), node_count) in enumerate(
        zip(stages, representatives, strict=True)
    ):
        for node_index in range(node_count):
            angle = 2 * np.pi * node_index / node_count
            points.append(
                (
                    float(stage_index),
                    float(np.cos(angle)),
                    float(np.sin(angle)),
                    f"{name}<br>{shape}<br>Channels: {channels}",
                    name if node_index == 0 else "",
                )
            )

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_z: list[float | None] = []
    offsets = np.cumsum([0, *representatives])
    for stage_index in range(len(stages) - 1):
        source = points[offsets[stage_index] : offsets[stage_index + 1]]
        target = points[offsets[stage_index + 1] : offsets[stage_index + 2]]
        for left in source:
            for right in target:
                edge_x.extend((left[0], right[0], None))
                edge_y.extend((left[1], right[1], None))
                edge_z.extend((left[2], right[2], None))

    colors = ["#62d7ff", "#9e8cff", "#62d7ff", "#9e8cff", "#62d7ff", "#ffbd69"]
    figure = go.Figure(
        [
            go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode="lines",
                line={"color": "rgba(139,165,197,.25)", "width": 2},
                hoverinfo="skip",
                showlegend=False,
            ),
            go.Scatter3d(
                x=[point[0] for point in points],
                y=[point[1] for point in points],
                z=[point[2] for point in points],
                mode="markers+text",
                text=[point[4] for point in points],
                textposition="top center",
                customdata=[point[3] for point in points],
                marker={
                    "size": 11,
                    "color": [
                        colors[index]
                        for index, count in enumerate(representatives)
                        for _ in range(count)
                    ],
                    "line": {"color": "white", "width": 1},
                },
                hovertemplate="%{customdata}<extra></extra>",
            ),
        ]
    )
    figure.update_layout(
        title="Inside the two-stage CIFAR-10 network",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=640,
        margin={"l": 0, "r": 0, "t": 55, "b": 0},
        scene={
            "xaxis": {
                "title": "Network depth",
                "tickmode": "array",
                "tickvals": list(range(len(stages))),
                "ticktext": [stage[0] for stage in stages],
            },
            "yaxis_title": "Channel arrangement",
            "zaxis_title": "Feature-channel group",
            "aspectmode": "auto",
            "camera": {"eye": {"x": 1.65, "y": 1.65, "z": 1.25}},
        },
    )
    return figure