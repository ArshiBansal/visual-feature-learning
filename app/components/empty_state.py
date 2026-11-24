"""Helpful empty states for pages gated by generated artifacts."""

import streamlit as st


def show_empty_state(
    title: str,
    description: str,
    *,
    command: str | None = None,
) -> None:
    """Explain a missing prerequisite and, when possible, show how to create it."""
    st.info(f"**{title}**\n\n{description}")
    if command:
        st.code(command, language="powershell")