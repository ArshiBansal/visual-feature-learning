"""Streamlit page for final model results and analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = PROJECT_ROOT / "outputs" / "cifar10_baseline" / "final_metrics.json"
ABLATION_RESULTS = PROJECT_ROOT / "outputs" / "ablation_studies" / "results.json"

st.set_page_config(page_title="Results & Analysis", page_icon="📊", layout="wide")
st.title("Results and Analysis")

if not METRICS_PATH.is_file():
    st.warning("Run evaluation to create `outputs/cifar10_baseline/final_metrics.json`.")
    st.stop()

with METRICS_PATH.open(encoding="utf-8") as file:
    metrics = json.load(file)

accuracy = float(metrics["test_accuracy"])
loss = float(metrics["test_loss"])

metric_columns = st.columns(4)
metric_columns[0].metric("Test accuracy", f"{accuracy:.2%}")
metric_columns[1].metric("Test loss", f"{loss:.4f}")
metric_columns[2].metric("Test samples", metrics["test_samples"])
metric_columns[3].metric("Parameters", f"{metrics['parameter_count']:,}")

confusion = pd.DataFrame(metrics["confusion_matrix"])
per_class = metrics["per_class_accuracy"]

left, right = st.columns(2)

with left:
    st.subheader("Confusion matrix")
    fig = go.Figure(go.Heatmap(
        z=confusion.values,
        colorscale="Blues",
        colorbar={"title": "Samples"},
    ))
    fig.update_layout(
        xaxis_title="Predicted class",
        yaxis_title="True class",
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Per-class accuracy")
    names = list(per_class)
    values = [float(value) * 100 for value in per_class.values()]
    fig = go.Figure(go.Bar(x=names, y=values))
    fig.update_layout(
        yaxis_title="Accuracy (%)",
        yaxis={"range": [0, 100]},
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Ablation summary")
if not ABLATION_RESULTS.is_file():
    st.info("No ablation results found.")
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

    st.dataframe(df, use_container_width=True)

st.subheader("Per-class metrics")
st.dataframe(
    pd.DataFrame({
        "Class": list(per_class),
        "Accuracy": [float(value) for value in per_class.values()],
    }),
    use_container_width=True,
)