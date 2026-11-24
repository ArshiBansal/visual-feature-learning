"""Compact artifact readiness panel for the Control Room."""

import streamlit as st

from app.services.artifacts import ArtifactPaths, metrics_are_current


def render_artifact_status(paths: ArtifactPaths) -> None:
    """Render the current state of training and evaluation artifacts."""
    for column, (label, path) in zip(
        st.columns(3),
        [
            ("Training history", paths.training_history),
            ("Learned checkpoint", paths.checkpoint),
            ("Evaluation report", paths.metrics),
        ],
        strict=True,
    ):
        with column:
            with st.container(border=True):
                st.markdown(f"**{label}**")
                if label == "Evaluation report" and paths.metrics.is_file():
                    if metrics_are_current(paths):
                        st.success("Current")
                    else:
                        st.warning("Stale — evaluate the latest checkpoint")
                elif path.is_file():
                    st.success("Ready")
                else:
                    st.caption("Not generated yet")