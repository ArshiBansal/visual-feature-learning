"""Streamlit page for learned filters and feature maps."""

from __future__ import annotations

import json
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

CONFIG_PATH = PROJECT_ROOT / "configs" / "cifar10_baseline.json"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "cifar10_baseline" / "cifar10_cnn_latest.npz"

st.set_page_config(page_title="Explore Feature Maps", page_icon="🔍", layout="wide")
st.title("Explore Feature Maps")

with CONFIG_PATH.open(encoding="utf-8") as file:
    config = json.load(file)

class_names = config["dataset"]["class_names"]

if not CHECKPOINT_PATH.is_file():
    st.warning("Run baseline training before exploring learned filters.")
    st.stop()

with np.load(CHECKPOINT_PATH) as checkpoint:
    conv1_weights = checkpoint["conv1_weights"].copy()
    conv2_weights = checkpoint["conv2_weights"].copy()

filter_tab, activation_tab = st.tabs(["Learned filters", "Activation maps"])

with filter_tab:
    columns = st.slider("Columns", 2, 8, 4)
    count = conv1_weights.shape[0]
    rows = (count + columns - 1) // columns

    if count:
        fig = go.Figure()
        for output_index in range(min(count, columns * rows)):
            rgb_filter = conv1_weights[output_index].transpose(1, 2, 0)
            low = rgb_filter.min()
            high = rgb_filter.max()
            display_filter = (rgb_filter - low) / (high - low + 1e-8)
            fig.add_trace(go.Image(z=np.clip(display_filter * 255, 0, 255).astype(np.uint8)))

        st.markdown("**First-layer RGB filters**")
        grid_columns = st.columns(columns)
        for index, ax_column in enumerate(grid_columns):
            for filter_index in range(index, count, columns):
                rgb_filter = conv1_weights[filter_index].transpose(1, 2, 0)
                low = rgb_filter.min()
                high = rgb_filter.max()
                display_filter = (rgb_filter - low) / (high - low + 1e-8)
                ax_column.image(display_filter, caption=f"Filter {filter_index}", use_container_width=True)

    channel_strength = np.sqrt(np.mean(conv2_weights**2, axis=(2, 3)))
    st.plotly_chart(go.Figure(go.Heatmap(
        z=channel_strength,
        x=[str(index) for index in range(channel_strength.shape[1])],
        y=[str(index) for index in range(channel_strength.shape[0])],
        colorscale="Viridis",
        colorbar={"title": "RMS weight"},
    )).update_layout(
        title="Second-layer filter strength by input channel",
        xaxis_title="Input feature-map channel",
        yaxis_title="Output filter",
    ), use_container_width=True)

with activation_tab:
    (_, _), (test_images, y_test) = load_cifar10()
    test_images = preprocess_cifar10_nchw(test_images)

    sample_count = min(len(test_images), 100)
    sample_index = st.slider("Test image", 0, sample_count - 1, 0)

    conv1_bias = np.load(CHECKPOINT_PATH)["conv1_bias"]
    conv2_bias = np.load(CHECKPOINT_PATH)["conv2_bias"]

    image = test_images[sample_index : sample_index + 1]

    st.caption(
        f"Actual class: {class_names[int(y_test[sample_index])]}"
    )

    conv1_output = np.zeros((1, conv1_weights.shape[0], 32, 32), dtype=np.float32)
    padded = np.pad(image, ((0, 0), (0, 0), (1, 1), (1, 1)))
    for output_index in range(conv1_weights.shape[0]):
        for kernel_y in range(3):
            for kernel_x in range(3):
                conv1_output[:, output_index] += padded[:, :, kernel_y:kernel_y + 32, kernel_x:kernel_x + 32] @ conv1_weights[output_index, :, kernel_y, kernel_x].reshape(1, -1) if False else np.einsum(
                    "nchw,chw->nhw",
                    padded[:, :, kernel_y:kernel_y + 32, kernel_x:kernel_x + 32],
                    conv1_weights[output_index:output_index + 1, :, kernel_y, kernel_x],
                )
        conv1_output[:, output_index] += conv1_bias[output_index]

    conv1_relu = np.maximum(conv1_output, 0)

    st.info("Add the packaged Conv2D/MaxPool2D modules to enable complete activation-map inference.")