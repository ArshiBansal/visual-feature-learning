"""Training Lab for launching the baseline and inspecting its trajectory."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import streamlit as st
from app.components.page_header import render_page_header
from app.services.artifacts import artifact_paths
from app.services.training_runner import (
    normalize_training_history,
    read_training_log,
    start_training,
)

from visualization.training_3d import plot_training_trajectory_3d

PROJECT_ROOT = Path(__file__).resolve().parents[2]
paths = artifact_paths(PROJECT_ROOT)
with paths.config.open(encoding="utf-8") as file:
    config = json.load(file)

training = config["training"]
optimizer = training["optimizer"]
process = st.session_state.get("training_process")
training_running = isinstance(process, subprocess.Popen) and process.poll() is None

render_page_header(
    "02 / BUILD THE MODEL",
    "Training Lab",
    "Launch the baseline and watch the model move through loss and accuracy. "
    "The 3D trajectory turns each epoch into a point in its learning journey.",
)

recipe = st.columns(4)
recipe[0].metric("Epochs", training["epochs"])
recipe[1].metric("Batch size", training["batch_size"])
recipe[2].metric("Learning rate", optimizer["learning_rate"])
recipe[3].metric("Optimizer", optimizer["name"].upper())

with st.expander("Full training recipe"):
    st.write(
        f"Momentum **{optimizer['momentum']}** · weight decay "
        f"**{optimizer['weight_decay']}** · schedule "
        f"**{training['learning_rate_schedule']['name']}**"
    )
    st.caption(
        "The run uses the configured CIFAR-10 training split. History and "
        "checkpoints are saved under `outputs/` and `checkpoints/`."
    )

action, status = st.columns([1, 2])
with action:
    start_clicked = st.button(
        "▶  Start baseline run",
        type="primary",
        disabled=training_running,
        use_container_width=True,
    )
with status:
    if training_running:
        st.info("Training is running. Refresh the page to update the live output.")
    elif process is not None and process.returncode == 0:
        st.success("The most recent run completed.")
    elif process is not None:
        st.error(f"The most recent run exited with code {process.returncode}.")
    else:
        st.caption("No training process has been started in this session.")

if start_clicked:
    try:
        process = start_training(PROJECT_ROOT, paths.training_log)
        st.session_state["training_process"] = process
        st.success("Training started. Refresh this page to check progress.")
    except OSError as error:
        st.error(f"Could not start training: {error}")

if paths.training_history.is_file():
    try:
        with paths.training_history.open(encoding="utf-8") as file:
            history = normalize_training_history(json.load(file))
    except (OSError, json.JSONDecodeError, ValueError) as error:
        history = None
        st.error(f"Could not read training history: {error}")

    if history:
        latest = history[-1]
        progress = st.columns(3)
        progress[0].metric("Latest epoch", f"{latest['epoch']} / {training['epochs']}")
        progress[1].metric("Training loss", f"{float(latest['train_loss']):.4f}")
        progress[2].metric("Training accuracy", f"{float(latest['train_accuracy']):.1%}")

        st.subheader("Learning trajectory")
        st.caption(
            "Rotate the 3D path to compare loss against accuracy. Color follows "
            "the order of the epochs."
        )
        st.plotly_chart(
            plot_training_trajectory_3d(history),
            use_container_width=True,
            config={"displaylogo": False, "scrollZoom": True},
        )
        with st.expander("Exact epoch metrics"):
            st.dataframe(history, use_container_width=True, hide_index=True)
    elif history is not None:
        st.warning("The training history file is empty. Run training to populate it.")
else:
    st.info(
        "Your first run will create the training history and model checkpoint "
        "used by the other lab pages."
    )

if paths.training_log.is_file():
    with st.expander("Training console output", expanded=training_running):
        log_text = read_training_log(paths.training_log)
        st.code(log_text or "Waiting for training output…", language="text")