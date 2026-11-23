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
from layers.conv2d import conv2d_forward
from layers.max_pool2d import max_pool2d_forward
from models.config import build_cifar10_cnn

CONFIG_PATH = PROJECT_ROOT / "configs" / "cifar10_baseline.json"
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "cifar10_baseline"
    / "cifar10_cnn_latest.npz"
)


def _load_checkpoint(
    path: Path,
    expected_shapes: dict[str, tuple[int, ...]],
) -> dict[str, np.ndarray]:
    loaded: dict[str, np.ndarray] = {}

    with np.load(path, allow_pickle=False) as checkpoint:
        for name, expected_shape in expected_shapes.items():
            if name not in checkpoint.files:
                raise ValueError(f"Checkpoint is missing {name!r}")

            array = np.asarray(checkpoint[name])
            if array.shape != expected_shape:
                if (
                    len(expected_shape) == 4
                    and array.ndim == 2
                    and array.size == int(np.prod(expected_shape))
                ):
                    array = array.reshape(expected_shape)
                else:
                    raise ValueError(
                        f"Checkpoint array {name!r} has shape {array.shape}; "
                        f"expected {expected_shape}"
                    )
            loaded[name] = array.astype(np.float32, copy=True)

    return loaded


@st.cache_data
def _load_test_data() -> tuple[np.ndarray, np.ndarray]:
    (_, _), (images, labels) = load_cifar10()
    return preprocess_cifar10_nchw(images), labels


st.set_page_config(
    page_title="Explore Feature Maps",
    page_icon="🔍",
    layout="wide",
)
st.title("Explore Feature Maps")

with CONFIG_PATH.open(encoding="utf-8") as file:
    config = json.load(file)

class_names = config["dataset"]["class_names"]
seed = int(config["project"]["seed"])

try:
    configured_model = build_cifar10_cnn(
        config,
        np.random.default_rng(seed),
    )
except (KeyError, TypeError, ValueError) as error:
    st.error(f"Invalid model configuration: {error}")
    st.stop()

if not CHECKPOINT_PATH.is_file():
    st.warning("Run baseline training before exploring learned filters.")
    st.stop()

expected_shapes = {
    "conv1_weights": configured_model.conv1.weights.shape,
    "conv1_bias": configured_model.conv1.bias.shape,
    "conv2_weights": configured_model.conv2.weights.shape,
    "conv2_bias": configured_model.conv2.bias.shape,
}

try:
    weights = _load_checkpoint(CHECKPOINT_PATH, expected_shapes)
except (OSError, ValueError) as error:
    st.error(f"Could not load the model checkpoint: {error}")
    st.stop()

conv1_weights = weights["conv1_weights"]
conv1_bias = weights["conv1_bias"]
conv2_weights = weights["conv2_weights"]
conv2_bias = weights["conv2_bias"]

filter_tab, activation_tab = st.tabs(["Learned filters", "Activation maps"])

with filter_tab:
    columns = st.slider("Columns", 2, 8, 4)
    filter_columns = st.columns(columns)

    for filter_index, kernel in enumerate(conv1_weights):
        rgb_filter = kernel.transpose(1, 2, 0)
        low = float(rgb_filter.min())
        high = float(rgb_filter.max())
        display_filter = (rgb_filter - low) / (high - low + 1e-8)

        filter_columns[filter_index % columns].image(
            display_filter,
            caption=f"Filter {filter_index}",
            use_container_width=True,
        )

    channel_strength = np.sqrt(np.mean(conv2_weights**2, axis=(2, 3)))
    strength_figure = go.Figure(
        go.Heatmap(
            z=channel_strength,
            x=[str(index) for index in range(channel_strength.shape[1])],
            y=[str(index) for index in range(channel_strength.shape[0])],
            colorscale="Viridis",
            colorbar={"title": "RMS weight"},
        )
    )
    strength_figure.update_layout(
        title="Second-layer filter strength by input channel",
        xaxis_title="Input feature-map channel",
        yaxis_title="Output filter",
    )
    st.plotly_chart(strength_figure, use_container_width=True)

with activation_tab:
    try:
        test_images, y_test = _load_test_data()
    except (ImportError, OSError, ValueError) as error:
        st.error(f"Could not load CIFAR-10 test data: {error}")
        st.stop()

    sample_index = st.slider(
        "Test image",
        0,
        len(test_images) - 1,
        0,
    )
    image = test_images[sample_index : sample_index + 1]

    st.caption(f"Actual class: {class_names[int(y_test[sample_index])]}")

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
        figure = go.Figure(
            go.Surface(
                z=first_map,
                x=np.arange(width),
                y=np.arange(height),
                colorscale="Viridis",
                colorbar={"title": "Activation"},
            )
        )
        figure.update_layout(
            title=f"First-stage activation channel {first_map_index}",
            height=650,
        )
        st.plotly_chart(figure, use_container_width=True)

    with right:
        second_map = pool2_output[0, second_map_index]
        height, width = second_map.shape
        figure = go.Figure(
            go.Surface(
                z=second_map,
                x=np.arange(width),
                y=np.arange(height),
                colorscale="Plasma",
                colorbar={"title": "Activation"},
            )
        )
        figure.update_layout(
            title=f"Second-stage activation channel {second_map_index}",
            height=650,
        )
        st.plotly_chart(figure, use_container_width=True)