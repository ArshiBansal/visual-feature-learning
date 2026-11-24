"""Page definitions and sidebar groups for the Streamlit learning lab."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

APP_ROOT = Path(__file__).resolve().parent
PAGES_ROOT = APP_ROOT / "pages"


def create_pages() -> tuple[dict[str, list[st.Page]], list[st.Page]]:
    """Create the grouped navigation tree and ordered page list."""
    page_specs = [
        ("00_Control_Room.py", "Control Room", "🧭"),
        ("01_Dataset_Atlas.py", "Dataset Atlas", "🌐"),
        ("02_Training_Lab.py", "Training Lab", "⚙️"),
        ("03_Feature_Studio.py", "Feature Studio", "🧬"),
        ("04_Model_Observatory.py", "Model Observatory", "🔭"),
        ("05_Experiment_Arena.py", "Experiment Arena", "🧪"),
    ]
    pages = [
        st.Page(
            str(PAGES_ROOT / filename),
            title=title,
            icon=icon,
            default=index == 0,
        )
        for index, (filename, title, icon) in enumerate(page_specs)
    ]
    return (
        {
            "Workspace": pages[:3],
            "Model intelligence": pages[3:],
        },
        pages,
    )