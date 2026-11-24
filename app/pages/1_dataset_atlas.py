"""Dataset Atlas for exploring CIFAR-10 in 3D color space."""

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

from data.cifar10 import load_cifar10  # noqa: E402
from visualization.dataset_3d import (  # noqa: E402
    plot_class_color_profiles_3d,
    plot_rgb_distribution_3d,
)

CONFIG_PATH = PROJECT_ROOT / "configs" / "cifar10_baseline.json"


@st.cache_data(show_spinner="Loading CIFAR-10 test samples…")
def _load_test_data() -> tuple[np.ndarray, np.ndarray]:
    return load_cifar10()[1]


render_page_header(
    "01 / UNDERSTAND THE DATA",
    "Dataset Atlas",
    "Explore CIFAR-10 without flattening it into a static gallery. Each image "
    "is represented in 3D color space, and each class has a measurable color signature.",
)

with CONFIG_PATH.open(encoding="utf-8") as file:
    config = json.load(file)
class_names = config["dataset"]["class_names"]

if not st.session_state.get("dataset_atlas_unlocked", False):
    with st.container(border=True):
        st.subheader("Bring the test split into view")
        st.write(
            "The atlas loads CIFAR-10 test samples on demand. This can download "
            "the dataset the first time, so nothing is fetched until you choose "
            "to open it."
        )
        if st.button("Load the CIFAR-10 atlas", type="primary"):
            st.session_state["dataset_atlas_unlocked"] = True
            st.rerun()
    st.stop()

try:
    test_images, test_labels = _load_test_data()
except (ImportError, OSError, ValueError) as error:
    show_empty_state(
        "CIFAR-10 data is not available",
        str(error),
        command='python -m pip install -e ".[datasets]"',
    )
    st.stop()

sample_count = min(1500, len(test_images))
rng = np.random.default_rng(int(config["project"]["seed"]))
sample_indices = np.sort(rng.choice(len(test_images), sample_count, replace=False))
sample_images = test_images[sample_indices]
sample_labels = test_labels[sample_indices]

metric_columns = st.columns(3)
metric_columns[0].metric("Official test split", f"{len(test_images):,}")
metric_columns[1].metric("Classes", len(class_names))
metric_columns[2].metric("Atlas sample", f"{sample_count:,}", "balanced, seeded subset")

class_distribution = np.bincount(sample_labels, minlength=len(class_names))
selected_class = st.selectbox("Choose a class", class_names)
selected_class_id = class_names.index(selected_class)
matching_indices = np.flatnonzero(sample_labels == selected_class_id)
if matching_indices.size == 0:
    st.warning(f"No {selected_class} samples were selected for this atlas view.")
    st.stop()

sample_number = st.slider(
    f"{selected_class.title()} sample in the atlas subset",
    min_value=0,
    max_value=int(matching_indices.size - 1),
    value=0,
)
image = sample_images[matching_indices[sample_number]]
global_sample_index = int(sample_indices[matching_indices[sample_number]])

left, right = st.columns([1.2, 1])
with left:
    st.subheader(f"{selected_class.title()} · sample {global_sample_index}")
    st.caption(
        "Each point is a pixel positioned by its red, green, and blue intensity. "
        "Hover to recover its original pixel coordinate and color."
    )
    st.plotly_chart(
        plot_rgb_distribution_3d(image),
        use_container_width=True,
        config={"displaylogo": False, "scrollZoom": True},
    )

with right:
    st.subheader("Class color signatures")
    st.caption(
        "A seeded 1,500-image test subset. Position encodes mean RGB; marker "
        "size reflects how many examples of each class appear in this subset."
    )
    st.plotly_chart(
        plot_class_color_profiles_3d(
            sample_images,
            sample_labels,
            class_names,
        ),
        use_container_width=True,
        config={"displaylogo": False, "scrollZoom": True},
    )

with st.expander("Exact subset counts by class"):
    st.dataframe(
        [
            {"Class": name, "Samples": int(class_distribution[index])}
            for index, name in enumerate(class_names)
        ],
        use_container_width=True,
        hide_index=True,
    )