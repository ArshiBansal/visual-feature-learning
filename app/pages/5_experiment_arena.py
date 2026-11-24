"""Experiment Arena for comparing ablation-study trade-offs."""

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
from app.services.artifacts import artifact_paths  # noqa: E402

from visualization.experiments_3d import plot_ablation_tradeoffs_3d  # noqa: E402

paths = artifact_paths(PROJECT_ROOT)
render_page_header(
    "05 / COMPARE IDEAS",
    "Experiment Arena",
    "See where each architecture lands when capacity, accuracy, and training "
    "cost compete. The best experiment depends on the trade-off you care about.",
)

if not paths.ablation_results.is_file():
    show_empty_state(
        "No ablation study has been recorded",
        "Run the ablation notebook and save experiment records to "
        "`outputs/ablation_studies/results.json`.",
    )
    st.stop()

with paths.ablation_results.open(encoding="utf-8") as file:
    results = json.load(file)
if not results:
    show_empty_state(
        "The experiment file is empty",
        "Add experiment entries before comparing architectures.",
    )
    st.stop()

results = sorted(results, key=lambda row: row["test_accuracy"], reverse=True)
best = results[0]
smallest = min(results, key=lambda row: row["parameter_count"])
fastest = min(results, key=lambda row: row["training_seconds"])

highlights = st.columns(3)
highlights[0].metric(
    "Top test accuracy",
    f"{float(best['test_accuracy']):.1%}",
    str(best["name"]),
)
highlights[1].metric(
    "Smallest model",
    f"{int(smallest['parameter_count']):,}",
    str(smallest["name"]),
)
highlights[2].metric(
    "Fastest experiment",
    f"{float(fastest['training_seconds']) / 60:.1f} min",
    str(fastest["name"]),
)

st.subheader("The trade-off landscape")
st.caption(
    "Point position encodes parameter count, test accuracy, and training time. "
    "Color shows test loss. Rotate to find the balance that matters to you."
)
st.plotly_chart(
    plot_ablation_tradeoffs_3d(results),
    use_container_width=True,
    config={"displaylogo": False, "scrollZoom": True},
)

experiment_table = pd.DataFrame(
    [
        {
            "Experiment": result["name"],
            "Test accuracy": float(result["test_accuracy"]),
            "Test loss": float(result["test_loss"]),
            "Parameters": int(result["parameter_count"]),
            "Training minutes": float(result["training_seconds"]) / 60,
        }
        for result in results
    ]
)
with st.expander("Exact experiment results"):
    st.dataframe(experiment_table, use_container_width=True, hide_index=True)