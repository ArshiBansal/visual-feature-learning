"""Streamlit page for training status and ablation comparisons."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "cifar10_baseline.json"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train.py"
HISTORY_PATH = PROJECT_ROOT / "outputs" / "cifar10_baseline" / "training_history.json"
TRAINING_LOG_PATH = PROJECT_ROOT / "outputs" / "cifar10_baseline" / "training.log"
ABLATION_RESULTS = PROJECT_ROOT / "outputs" / "ablation_studies" / "results.json"

st.set_page_config(page_title="Train & Compare", page_icon="🧠", layout="wide")
st.title("Train and Compare")
st.caption("Baseline training and ablation-study artifacts")

with CONFIG_PATH.open(encoding="utf-8") as file:
    config = json.load(file)

training = config["training"]
optimizer = training["optimizer"]

with st.sidebar:
    st.header("Baseline settings")
    st.metric("Epochs", training["epochs"])
    st.metric("Batch size", training["batch_size"])
    st.metric("Learning rate", optimizer["learning_rate"])
    st.metric("Momentum", optimizer["momentum"])
    st.metric("Weight decay", optimizer["weight_decay"])

process = st.session_state.get("training_process")
training_running = process is not None and process.poll() is None

baseline_tab, ablation_tab = st.tabs(
    ["Baseline training", "Ablation comparison"]
)

with baseline_tab:
    if not HISTORY_PATH.is_file():
        st.info("Run baseline training to create training history.")
    else:
        with HISTORY_PATH.open(encoding="utf-8") as file:
            history = json.load(file)

        df = pd.DataFrame(history)
        left, right = st.columns(2)

        with left:
            figure = go.Figure()
            figure.add_trace(
                go.Scatter(
                    x=df["epoch"],
                    y=df["train_loss"],
                    mode="lines+markers",
                    name="Loss",
                )
            )
            figure.update_layout(
                title="Training loss",
                xaxis_title="Epoch",
                yaxis_title="Loss",
            )
            st.plotly_chart(figure, use_container_width=True)

        with right:
            figure = go.Figure()
            figure.add_trace(
                go.Scatter(
                    x=df["epoch"],
                    y=df["train_accuracy"],
                    mode="lines+markers",
                    name="Accuracy",
                )
            )
            figure.update_layout(
                title="Training accuracy",
                xaxis_title="Epoch",
                yaxis_title="Accuracy",
            )
            st.plotly_chart(figure, use_container_width=True)

        st.dataframe(df, use_container_width=True)

    if training_running:
        st.info("Baseline training is running in the background.")
    elif process is not None:
        if process.returncode == 0:
            st.success("Baseline training completed.")
        else:
            st.error(
                f"Baseline training exited with code {process.returncode}."
            )

    start_column, refresh_column = st.columns(2)
    with start_column:
        if st.button(
            "Start baseline training",
            type="primary",
            disabled=training_running,
        ):
            TRAINING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            try:
                with TRAINING_LOG_PATH.open("w", encoding="utf-8") as log_file:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-u",
                            str(TRAIN_SCRIPT),
                        ],
                        cwd=PROJECT_ROOT,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                st.session_state["training_process"] = process
                st.success("Training started. Refresh to check progress.")
            except OSError as error:
                st.error(f"Could not start training: {error}")

    with refresh_column:
        st.button("Refresh training status")

    if TRAINING_LOG_PATH.is_file():
        log_text = TRAINING_LOG_PATH.read_text(
            encoding="utf-8",
            errors="replace",
        )
        st.text_area(
            "Training log",
            value=log_text[-12000:],
            height=250,
            disabled=True,
        )

with ablation_tab:
    if not ABLATION_RESULTS.is_file():
        st.info(
            "Run the ablation notebook to create "
            "`outputs/ablation_studies/results.json`."
        )
    else:
        with ABLATION_RESULTS.open(encoding="utf-8") as file:
            results = json.load(file)

        df = pd.DataFrame(
            [
                {
                    "Experiment": result["name"],
                    "Test accuracy": result["test_accuracy"],
                    "Test loss": result["test_loss"],
                    "Parameters": result["parameter_count"],
                    "Training minutes": result["training_seconds"] / 60,
                }
                for result in results
            ]
        ).sort_values("Test accuracy", ascending=False)

        left, right = st.columns(2)
        with left:
            figure = go.Figure(
                go.Bar(
                    x=df["Test accuracy"] * 100,
                    y=df["Experiment"],
                    orientation="h",
                )
            )
            figure.update_layout(
                title="Test accuracy",
                xaxis_title="Accuracy (%)",
                yaxis_title="",
            )
            st.plotly_chart(figure, use_container_width=True)

        with right:
            figure = go.Figure(
                go.Bar(
                    x=df["Parameters"],
                    y=df["Experiment"],
                    orientation="h",
                )
            )
            figure.update_layout(
                title="Model size",
                xaxis_title="Parameters",
                yaxis_title="",
            )
            st.plotly_chart(figure, use_container_width=True)

        figure = go.Figure(
            go.Scatter3d(
                x=df["Parameters"],
                y=df["Test accuracy"],
                z=df["Training minutes"],
                mode="markers",
                text=df["Experiment"],
                marker={"size": 7, "color": df["Test accuracy"]},
            )
        )
        figure.update_layout(
            title="Accuracy, capacity, and training cost"
        )
        st.plotly_chart(figure, use_container_width=True)

        st.dataframe(df, use_container_width=True)