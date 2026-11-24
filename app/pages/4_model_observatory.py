"""Model Observatory for held-out performance by class."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from app.components.empty_state import show_empty_state  # noqa: E402
from app.components.page_header import render_page_header  # noqa: E402
from app.services.artifacts import artifact_paths, metrics_are_current  # noqa: E402

from visualization.evaluation_3d import (  # noqa: E402
    plot_confusion_matrix_3d,
    plot_per_class_accuracy_3d,
)

paths = artifact_paths(PROJECT_ROOT)
render_page_header(
    "04 / MEASURE GENERALIZATION",
    "Model Observatory",
    "Explore what the model gets right, which classes it confuses, and how "
    "consistently it performs across the held-out test set.",
)

if not paths.metrics.is_file():
    show_empty_state(
        "Evaluation results are not available",
        "Train a checkpoint first, then evaluate it to populate the class-level views.",
        command="python scripts/evaluate.py",
    )
    st.stop()
if not metrics_are_current(paths):
    st.warning(
        "These metrics are older than the current checkpoint. Rerun evaluation "
        "before treating the results as current."
    )

with paths.metrics.open(encoding="utf-8") as file:
    metrics = json.load(file)

accuracy = float(metrics["test_accuracy"])
loss = float(metrics["test_loss"])
class_names = list(metrics["per_class_accuracy"])
confusion = pd.DataFrame(metrics["confusion_matrix"])
if confusion.shape != (len(class_names), len(class_names)):
    st.error("The confusion matrix and per-class metrics have incompatible shapes.")
    st.stop()

overview = st.columns(4)
overview[0].metric("Test accuracy", f"{accuracy:.2%}")
overview[1].metric("Test loss", f"{loss:.4f}")
overview[2].metric("Test samples", f"{int(metrics['test_samples']):,}")
overview[3].metric("Parameters", f"{int(metrics['parameter_count']):,}")

st.subheader("Where the model gets confused")
st.caption(
    "Each point is an actual/predicted class pair. Height and color show the "
    "number of test examples; hover for the exact class names and count."
)
st.plotly_chart(
    plot_confusion_matrix_3d(confusion.to_numpy(), class_names),
    use_container_width=True,
    config={"displaylogo": False, "scrollZoom": True},
)

st.subheader("Accuracy by class")
st.caption(
    "Compare class recall against its test-set support. The exact-value table "
    "below keeps the 3D view interpretable."
)
st.plotly_chart(
    plot_per_class_accuracy_3d(confusion.to_numpy(), class_names),
    use_container_width=True,
    config={"displaylogo": False, "scrollZoom": True},
)

class_table = pd.DataFrame(
    {
        "Class": class_names,
        "Accuracy": [float(metrics["per_class_accuracy"][name]) for name in class_names],
        "Test samples": confusion.sum(axis=1).astype(int),
    }
)
with st.expander("Exact class metrics and confusion counts"):
    st.dataframe(class_table, use_container_width=True, hide_index=True)
    confusion.index = class_names
    confusion.columns = class_names
    st.dataframe(confusion, use_container_width=True)