"""Shared theme helpers for the dark UI."""

ACCENT_PRIMARY = "#00A8FF"   # electric blue
ACCENT_SECONDARY = "#00E5FF" # cyan
BG_DARK = "#0B1120"          # dark navy
BG_PANEL = "#131C2E"
TEXT_MAIN = "#E6EDF5"
TEXT_MUTED = "#8FA3BF"

CUSTOM_CSS = f"""
<style>
    :root {{
        --accent: {ACCENT_PRIMARY};
        --accent-2: {ACCENT_SECONDARY};
        --bg: {BG_DARK};
        --panel: {BG_PANEL};
        --text: {TEXT_MAIN};
        --muted: {TEXT_MUTED};
    }}
    .stApp {{
        background-color: var(--bg);
        color: var(--text);
    }}
    h1, h2, h3 {{ color: var(--text); letter-spacing: 0.2px; }}
    .brand-title {{
        font-size: 2.4rem; font-weight: 700;
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .brand-sub {{ color: var(--muted); font-size: 1.05rem; margin-top: -0.4rem; }}
    div[data-testid="stSidebar"] {{
        background-color: #0A0F1B;
        border-right: 1px solid rgba(0,168,255,0.18);
    }}
    .stButton > button {{
        border: 1px solid var(--accent);
        color: var(--accent-2);
        border-radius: 8px;
    }}
    .stButton > button[kind="primary"] {{
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
        color: #04101F; font-weight: 600; border: none;
    }}
    .panel {{
        background: var(--panel);
        border: 1px solid rgba(0,168,255,0.18);
        border-radius: 12px;
        padding: 1rem 1.25rem;
    }}
    .status-pill {{
        display: inline-block; padding: 2px 12px; border-radius: 999px;
        font-size: 0.85rem; border: 1px solid var(--accent); color: var(--accent-2);
    }}
</style>
"""


def apply_theme() -> None:
    import streamlit as st
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
