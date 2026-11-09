"""Settings page — application info, storage, provider & credential status."""

import streamlit as st

from .. import config, db
from ..providers import get_registry


def _pill(ok: bool) -> str:
    return "🟢 Configured" if ok else "🔴 Not configured"


def render() -> None:
    st.markdown('<div class="brand-title">Settings</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">Application information and configuration status.'
                "</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Application")
        st.markdown(f"- **Name:** {config.APP_NAME}")
        st.markdown(f"- **Version:** {config.APP_VERSION}")
        st.markdown("- **Interface:** Streamlit (local)")
        st.markdown("- **Database:** SQLite")

    with col_b:
        st.markdown("#### Storage")
        st.markdown(f"- **Data directory:** `{config.DATA_DIR}`")
        st.markdown(f"- **Database file:** `{config.DB_PATH}`")
        st.markdown(f"- **Voice samples:** `{config.SAMPLES_DIR}`")
        st.markdown(f"- **Generated audio:** `{config.GENERATED_DIR}`")
        st.markdown(f"- **Max upload size:** {config.MAX_UPLOAD_MB} MB")

    st.markdown("#### Provider status")
    registry = get_registry()
    for provider in registry.list_providers():
        caps = provider.get_capabilities()
        with st.expander(f"⚙️ {caps.name}", expanded=False):
            st.markdown(f"- Type: {'Cloud' if caps.supports_cloud else 'Local'}")
            st.markdown(f"- Credentials/config: {_pill(caps.configured)}")
            st.markdown(f"- Generation: {'✅ ready' if caps.generation_ready else '🔒 not yet implemented'}")
            st.markdown(f"- Languages: {', '.join(caps.languages)}")
            if caps.notes:
                st.caption(caps.notes)
            missing = provider.missing_configuration()
            if missing:
                st.warning("Missing: " + "; ".join(missing))

    st.markdown("#### API credential status")
    st.markdown("- ElevenLabs: " + _pill(bool(config.ELEVENLABS_API_KEY)))
    st.caption("API keys are never displayed or written to logs — only their presence is shown.")

    st.markdown("#### Audio export preferences")
    fmt = st.selectbox("Default export format", ["WAV", "MP3"],
                       index=0 if db.get_setting("export_format", "wav") == "wav" else 1)
    if st.button("Save preferences"):
        db.set_setting("export_format", fmt.lower())
        st.success("Preferences saved locally.")
