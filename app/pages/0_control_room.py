"""Control Room for model overview, status, and guided next steps."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from app.components.page_header import render_page_header  # noqa: E402
from app.components.status_panel import render_artifact_status  # noqa: E402
from app.services.artifacts import artifact_paths  # noqa: E402

from visualization.architecture_3d import plot_model_architecture_3d  # noqa: E402

paths = artifact_paths(PROJECT_ROOT)
with paths.config.open(encoding="utf-8") as file:
    config = json.load(file)

render_page_header(
    "WELCOME / YOUR WORKSPACE",
    "Control Room",
    "A hands-on lab for seeing how a NumPy convolutional network learns, "
    "what it notices, and where it still gets confused.",
)

st.markdown(
    """
    <div class="lab-hero">
      <div class="lab-eyebrow">From pixels to predictions</div>
      <h1>Make the model<br>make sense.</h1>
      <p>
        Explore CIFAR-10 through a complete learning journey. Start training,
        follow feature maps as they emerge, then challenge the model with
        held-out images and class-level results.
      </p>
      <span class="lab-pill">PURE NUMPY</span>
      <span class="lab-pill">10 CLASSES</span>
      <span class="lab-pill">INTERACTIVE 3D</span>
    </div>
    """,
    unsafe_allow_html=True,
)

overview = st.columns(4)
overview[0].metric("Training images", f"{config['dataset']['expected_train_samples']:,}")
overview[1].metric("Test images", f"{config['dataset']['expected_test_samples']:,}")
overview[2].metric("Network", "2 conv stages")
overview[3].metric("Class labels", config["dataset"]["num_classes"])

st.subheader("The model, made visible")
st.caption(
    "Follow the tensor shapes through the real configured CNN. Hover on nodes "
    "for details, and rotate the graph to understand its depth."
)
st.plotly_chart(
    plot_model_architecture_3d(),
    use_container_width=True,
    config={"displaylogo": False, "scrollZoom": True},
)

st.subheader("Lab readiness")
render_artifact_status(paths)

if not paths.checkpoint.is_file():
    st.info(
        "Start in **Training Lab**. A completed training run unlocks the "
        "Feature Studio; evaluation unlocks the Model Observatory."
    )
elif not paths.metrics.is_file():
    st.info("A checkpoint is ready. Run evaluation to unlock test-set analysis.")
else:
    st.success("Your model and evaluation artifacts are ready to explore.")