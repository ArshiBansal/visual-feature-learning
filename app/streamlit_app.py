"""Streamlit entry point for the visual feature learning project."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "cifar10_baseline.json"
METRICS_PATH = PROJECT_ROOT / "outputs" / "cifar10_baseline" / "final_metrics.json"
HISTORY_PATH = PROJECT_ROOT / "outputs" / "cifar10_baseline" / "training_history.json"
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "cifar10_baseline"
    / "cifar10_cnn_latest.npz"
)


def _metrics_are_current() -> bool:
    if not METRICS_PATH.is_file() or not CHECKPOINT_PATH.is_file():
        return False

    return METRICS_PATH.stat().st_mtime_ns >= CHECKPOINT_PATH.stat().st_mtime_ns


st.set_page_config(
    page_title="Visual Feature Learning",
    page_icon="🧠",
    layout="wide",
)

st.title("Visual Feature Learning")
st.caption("Pure-NumPy CIFAR-10 convolutional neural network")

with CONFIG_PATH.open(encoding="utf-8") as file:
    config = json.load(file)

left, center, right = st.columns(3)
left.metric("Dataset", "CIFAR-10")
center.metric("Training samples", f"{config['dataset']['expected_train_samples']:,}")
right.metric("Test samples", f"{config['dataset']['expected_test_samples']:,}")

st.divider()

overview_left, overview_right = st.columns(2)

with overview_left:
    st.subheader("Project overview")
    st.markdown(
        """
        This project implements and inspects a two-stage convolutional neural
        network using NumPy. It uses the official CIFAR-10 training and test
        splits, NCHW image tensors, and a pure NumPy forward/backward pass.

        Use the sidebar to open:
        - **Train and Compare** for baseline training and ablation results
        - **Explore Feature Maps** for learned filters and activations
        - **Results and Analysis** for test metrics and the confusion matrix
        """
    )

with overview_right:
    st.subheader("Artifact status")
    st.write(
        "Training history:",
        "ready" if HISTORY_PATH.is_file() else "not generated",
    )
    st.write(
        "Latest checkpoint:",
        "ready" if CHECKPOINT_PATH.is_file() else "not generated",
    )

    metrics_current = _metrics_are_current()
    if metrics_current:
        metrics_status = "current"
    elif METRICS_PATH.is_file():
        metrics_status = "stale — rerun evaluation"
    else:
        metrics_status = "not generated"

    st.write("Final metrics:", metrics_status)

st.divider()

if _metrics_are_current():
    with METRICS_PATH.open(encoding="utf-8") as file:
        metrics = json.load(file)

    st.subheader("Latest evaluated checkpoint metrics")
    metric_columns = st.columns(3)
    metric_columns[0].metric(
        "Test accuracy",
        f"{float(metrics['test_accuracy']):.2%}",
    )
    metric_columns[1].metric(
        "Test loss",
        f"{float(metrics['test_loss']):.4f}",
    )
    metric_columns[2].metric(
        "Parameters",
        f"{int(metrics['parameter_count']):,}",
    )
elif METRICS_PATH.is_file():
    st.warning(
        "The checkpoint has changed since these metrics were generated. "
        "Run `python scripts/evaluate.py` to refresh them."
    )
else:
    st.info(
        "Train and evaluate the baseline to populate final metrics. "
        "Run `python scripts/train.py`, then `python scripts/evaluate.py`."
    )

st.caption("Configuration: `configs/cifar10_baseline.json`")