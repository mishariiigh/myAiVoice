"""Entry point for Mishari Voice Studio (Streamlit)."""

import streamlit as st

from . import config, db
from .theme import apply_theme


def ensure_ready() -> None:
    config.ensure_directories()
    db.init_db()


def main() -> None:
    st.set_page_config(
        page_title="Mishari Voice Studio",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    ensure_ready()
    apply_theme()

    st.sidebar.markdown("### 🎙️ Mishari Voice Studio")
    page = st.sidebar.radio(
        "Navigation",
        ["Studio", "Voice Library", "Generation History", "Settings"],
        label_visibility="collapsed",
    )
    st.sidebar.caption(f"v{config.APP_VERSION}")

    from .pages import history, settings, studio, voice_library

    if page == "Studio":
        studio.render()
    elif page == "Voice Library":
        voice_library.render()
    elif page == "Generation History":
        history.render()
    else:
        settings.render()


if __name__ == "__main__":
    main()
