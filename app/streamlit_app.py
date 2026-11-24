"""Entrypoint and branded navigation shell for the visual learning lab."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (APP_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from navigation import create_pages  # noqa: E402

st.set_page_config(
    page_title="Visual Learning Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

theme_css = (APP_ROOT / "theme" / "styles.css").read_text(encoding="utf-8")
st.markdown(f"<style>{theme_css}</style>", unsafe_allow_html=True)

navigation_groups, all_pages = create_pages()
with st.sidebar:
    st.markdown(
        """
        <div style="padding:.4rem .2rem 1.1rem">
          <div style="font-size:1.55rem;font-weight:800;color:#edf4ff">
            ◈ VISUAL LEARNING LAB
          </div>
          <div style="font-size:.78rem;color:#9aacc5;margin-top:.25rem">
            CIFAR-10 · NUMPY CNN
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for group, pages in navigation_groups.items():
        st.caption(group.upper())
        for page in pages:
            st.page_link(page, label=page.title, icon=page.icon)
    st.divider()
    st.markdown("**A model you can see inside.**")
    st.caption(
        "Train from scratch, follow learned features, and inspect every result "
        "through interactive 3D views."
    )

active_page = st.navigation(navigation_groups, position="hidden")
active_page.run()