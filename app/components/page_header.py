"""Consistent page title treatment for the visual learning lab."""

import streamlit as st


def render_page_header(eyebrow: str, title: str, description: str) -> None:
    """Render a concise section marker, title, and page introduction."""
    st.markdown(
        f'<div class="lab-eyebrow">{eyebrow}</div>',
        unsafe_allow_html=True,
    )
    st.title(title)
    st.markdown(description)