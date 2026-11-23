"""Streamlit page for learned filters and feature maps."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.cifar10 import load_cifar10
from data.preprocessing import preprocess_cifar10_nchw
from layers.conv2d import conv2d_forward
from layers.max_pool2d import max_pool2d_forward

CONFIG_PATH = PROJECT_ROOT / "configs" / "cifar10_baseline.json"
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "cifar10_baseline"
    / "cifar10_cnn_latest.npz"
)

st.set_page_config(
    page_title="Explore Feature Maps",
    page_icon="🔍",
    layout="wide",
)
st.title("Explore Feature Maps")

with (CONFIG_PATH).open(encoding="utf-8") as file:
    import json
    config = json.load(file)

class_names = config["dataset"]["class_names"]

if not CHECKPOINT_PATH.is_file():
    st.warning("Run baseline training before exploring learned filters.")
    st.stop()

with np.load(CHECKPOINT_PATH) as checkpoint:
    conv1_weights = checkpoint["conv1_weights"].copy()
    conv1_bias = checkpoint["conv1_bias"].copy()
    conv2_weights = checkpoint["conv2_weights"].copy()
    conv2_bias = checkpoint["conv2_bias"].copy()

filter_tab, activation_tab = st.tabs(["Learned filters", "Activation maps"])

with filter_tab:
    columns = st.slider("Columns", 2, 8, 4)
    count = conv1_weights.shape[0]

    if count:
        grid_columns = st.columns(columns)
        for filter_index in range(count):
            rgb_filter = conv1_weights[filter_index].transpose(1, 2, 0)
            low = float(rgb_filter.min())
            high = float(rgb_filter.max())
            display_filter = (rgb_filter - low) / (high - low + 1e-8)

            column = grid_columns[filter_index % columns]
            column.image(
                display_filter,
                caption=f"Filter {filter_index}",
                use_container_width=True,
            )

    channel_strength = np.sqrt(
        np.mean(conv2_weights**2, axis=(2, 3))
    )
    st.plotly_chart(
        go.Figure(
            go.Heatmap(
                z=channel_strength,
                x=[
                    str(index)
                    for index in range(channel_strength.shape[1])
                ],
                y=[
                    str(index)
                    for index in range(channel_strength.shape[0])
                ],
                colorscale="Viridis",
                colorbar={"title": "RMS weight"},
            )
        ).update_layout(
            title="Second-layer filter strength by input channel",
            xaxis_title="Input feature-map channel",
            yaxis_title="Output filter",
        ),
        use_container_width=True,
    )

with activation_tab:
    (_, _), (test_images, y_test) = load_cifar10()
    test_images = preprocess_cifar10_nchw(test_images)

    sample_count = len(test_images)
    sample_index = st.slider(
        "Test image",
        0,
        sample_count - 1,
        0,
    )

    image = test_images[sample_index : sample_index + 1]

    st.caption(
        f"Actual class: {class_names[int(y_test[sample_index])]}"
    )

    conv1_output, _ = conv2d_forward(
        image,
        conv1_weights,
        conv1_bias,
        stride=1,
        padding="same",
    )
    conv1_relu = np.maximum(conv1_output, 0)
    pool1_output, _ = max_pool2d_forward(
        conv1_relu,
        pool_size=2,
        stride=2,
    )

    conv2_output, _ = conv2d_forward(
        pool1_output,
        conv2_weights,
        conv2_bias,
        stride=1,
        padding="same",
    )
    conv2_relu = np.maximum(conv2_output, 0)
    pool2_output, _ = max_pool2d_forward(
        conv2_relu,
        pool_size=2,
        stride=2,
    )

    first_map_index = st.slider(
        "First-stage activation channel",
        0,
        conv1_relu.shape[1] - 1,
        0,
    )
    second_map_index = st.slider(
        "Second-stage activation channel",
        0,
        pool2_output.shape[1] - 1,
        0,
    )

    left, right = st.columns(2)

    with left:
        first_map = conv1_relu[0, first_map_index]
        height, width = first_map.shape
        fig = go.Figure(
            go.Surface(
                z=first_map,
                x=np.arange(width),
                y=np.arange(height),
                colorscale="Viridis",
                colorbar={"title": "Activation"},
            )
        )
        fig.update_layout(
            title=f"First-stage activation channel {first_map_index}",
            height=650,
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        second_map = pool2_output[0, second_map_index]
        height, width = second_map.shape
        fig = go.Figure(
            go.Surface(
                z=second_map,
                x=np.arange(width),
                y=np.arange(height),
                colorscale="Plasma",
                colorbar={"title": "Activation"},
            )
        )
        fig.update_layout(
            title=f"Second-stage activation channel {second_map_index}",
            height=650,
        )
        st.plotly_chart(fig, use_container_width=True)