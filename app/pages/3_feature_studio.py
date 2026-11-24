"""Feature Studio for inspecting kernels, activations, and predictions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from app.components.empty_state import show_empty_state  # noqa: E402
from app.components.page_header import render_page_header  # noqa: E402
from app.services.artifacts import artifact_paths  # noqa: E402

from data.cifar10 import load_cifar10  # noqa: E402
from data.preprocessing import preprocess_cifar10_nchw  # noqa: E402
from inference.predict import predict_cifar10_image  # noqa: E402
from layers.conv2d import conv2d_forward  # noqa: E402
from layers.max_pool2d import max_pool2d_forward  # noqa: E402
from models.config import build_cifar10_cnn  # noqa: E402
from visualization.activations_3d import plot_activation_surface  # noqa: E402
from visualization.features_3d import (  # noqa: E402
    plot_filter_channel_strength_3d,
    plot_filter_kernel_3d,
)


@st.cache_data
def _load_test_data() -> tuple[np.ndarray, np.ndarray]:
    return load_cifar10()[1]


@st.cache_resource
def _load_model(checkpoint_path: str, checkpoint_mtime_ns: int, config_json: str):
    config = json.loads(config_json)
    checkpoint_file = Path(checkpoint_path)
    if checkpoint_file.stat().st_mtime_ns != checkpoint_mtime_ns:
        raise ValueError("Checkpoint changed while it was being loaded; refresh and retry.")
    model = build_cifar10_cnn(
        config,
        np.random.default_rng(int(config["project"]["seed"])),
    )
    with np.load(checkpoint_file, allow_pickle=False) as checkpoint:
        targets = {
            "conv1_weights": (model.conv1, "weights"),
            "conv1_bias": (model.conv1, "bias"),
            "conv2_weights": (model.conv2, "weights"),
            "conv2_bias": (model.conv2, "bias"),
            "dense_weights": (model.dense, "weights"),
            "dense_bias": (model.dense, "bias"),
        }
        for name, (layer, attribute) in targets.items():
            if name not in checkpoint.files:
                raise ValueError(f"Checkpoint is missing {name!r}")
            value = np.asarray(checkpoint[name])
            expected = getattr(layer, attribute).shape
            if value.shape != expected:
                raise ValueError(
                    f"Checkpoint array {name!r} has shape {value.shape}; expected {expected}"
                )
            setattr(layer, attribute, value.astype(np.float32, copy=True))
    return model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
paths = artifact_paths(PROJECT_ROOT)
render_page_header(
    "03 / OPEN THE MODEL",
    "Feature Studio",
    "Move from learned convolution weights to the responses they create. "
    "Pick a held-out image, inspect the model's prediction, and explore its "
    "3D activation surfaces.",
)

if not paths.checkpoint.is_file():
    show_empty_state(
        "No trained checkpoint yet",
        "Run the baseline in Training Lab to create the learned filters used here.",
    )
    st.stop()

with paths.config.open(encoding="utf-8") as file:
    config = json.load(file)
class_names = config["dataset"]["class_names"]

try:
    model = _load_model(
        str(paths.checkpoint),
        paths.checkpoint.stat().st_mtime_ns,
        json.dumps(config, sort_keys=True),
    )
except (OSError, ValueError, KeyError) as error:
    st.error(f"Could not load the configured model checkpoint: {error}")
    st.stop()

view = st.radio(
    "Choose what to inspect",
    ["Kernel microscope", "Sample journey"],
)
if view == "Kernel microscope":
    filter_index = st.slider(
        "First-layer filter",
        min_value=0,
        max_value=model.conv1.weights.shape[0] - 1,
        value=0,
    )
    st.caption(
        "Every point is a kernel weight positioned by spatial location and input "
        "color channel. Color shows the sign; size shows magnitude."
    )
    st.plotly_chart(
        plot_filter_kernel_3d(
            model.conv1.weights[filter_index],
            title=f"First-layer filter {filter_index}",
        ),
        use_container_width=True,
        config={"displaylogo": False, "scrollZoom": True},
    )

    st.subheader("Channel connections")
    st.caption(
        "RMS weight strength for each input/output channel pair in the second "
        "convolution. Height and color reveal stronger connections."
    )
    st.plotly_chart(
        plot_filter_channel_strength_3d(model.conv2.weights),
        use_container_width=True,
        config={"displaylogo": False, "scrollZoom": True},
    )

else:
    try:
        test_images, test_labels = _load_test_data()
    except (ImportError, OSError, ValueError) as error:
        show_empty_state(
            "CIFAR-10 test data is not available",
            str(error),
            command='python -m pip install -e ".[datasets]"',
        )
        st.stop()

    sample_index = st.slider(
        "Test image index",
        min_value=0,
        max_value=len(test_images) - 1,
        value=0,
    )
    image = test_images[sample_index]
    prediction = predict_cifar10_image(model, image, class_names)
    actual_class = class_names[int(test_labels[sample_index])]
    scores = st.columns(3)
    scores[0].metric("Actual class", actual_class)
    scores[1].metric("Model prediction", prediction.predicted_class)
    scores[2].metric("Confidence", f"{prediction.confidence:.1%}")
    if prediction.predicted_class == actual_class:
        st.success("The model classified this sample correctly.")
    else:
        st.warning(
            f"The model mistook this {actual_class} for a "
            f"{prediction.predicted_class}. Compare the activations below."
        )

    image_batch = preprocess_cifar10_nchw(image[None, ...])
    conv1, _ = conv2d_forward(
        image_batch,
        model.conv1.weights,
        model.conv1.bias,
        stride=1,
        padding="same",
    )
    conv1 = np.maximum(conv1, 0)
    pooled1, _ = max_pool2d_forward(conv1, pool_size=2, stride=2)
    conv2, _ = conv2d_forward(
        pooled1,
        model.conv2.weights,
        model.conv2.bias,
        stride=1,
        padding="same",
    )
    conv2 = np.maximum(conv2, 0)
    pooled2, _ = max_pool2d_forward(conv2, pool_size=2, stride=2)

    first_channel, second_channel = st.columns(2)
    first_index = first_channel.slider(
        "Early feature channel",
        min_value=0,
        max_value=conv1.shape[1] - 1,
        value=0,
    )
    second_index = second_channel.slider(
        "Deeper feature channel",
        min_value=0,
        max_value=pooled2.shape[1] - 1,
        value=0,
    )
    first_plot, second_plot = st.columns(2)
    with first_plot:
        st.markdown("#### Early color and edge response")
        st.plotly_chart(
            plot_activation_surface(
                conv1[0, first_index],
                title=f"Early feature channel {first_index}",
            ),
            use_container_width=True,
        )
    with second_plot:
        st.markdown("#### Deeper spatial response")
        st.plotly_chart(
            plot_activation_surface(
                pooled2[0, second_index],
                title=f"Deeper feature channel {second_index}",
                colorscale="Plasma",
            ),
            use_container_width=True,
        )