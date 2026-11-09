"""Studio page — script editor and generation controls (Stage 1: no real generation)."""

import streamlit as st

from .. import config, db
from ..providers import get_registry
from ..theme import ACCENT_PRIMARY, ACCENT_SECONDARY


def render() -> None:
    st.markdown('<div class="brand-title">Studio</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-sub">Compose a script, pick an engine and a voice '
        "profile, then generate speech in English or Arabic.</div>",
        unsafe_allow_html=True,
    )

    registry = get_registry()
    profiles = db.list_voice_profiles()

    col1, col2, col3 = st.columns(3)
    language = col1.selectbox("Language", list(config.SUPPORTED_LANGUAGES))

    cloud_providers = [p for p in registry.list_providers() if p.get_capabilities().supports_cloud]
    local_providers = [p for p in registry.list_providers() if p.get_capabilities().supports_local]
    category = col2.radio("Provider", ["Cloud", "Local"], horizontal=True)
    candidates = cloud_providers if category == "Cloud" else local_providers
    provider_options = {
        p.get_capabilities().name: p for p in candidates
    } or {"No provider available": None}
    chosen_name = col3.selectbox("Engine", list(provider_options.keys()))
    provider = provider_options[chosen_name]

    profile_names = [p["name"] for p in profiles] or ["No voice profiles yet"]
    profile_name = st.selectbox("Voice profile", profile_names)
    selected_profile = next((p for p in profiles if p["name"] == profile_name), None)

    script = st.text_area(
        "Script",
        height=220,
        placeholder="Type or paste the text you want to speak…",
    )

    caps = provider.get_capabilities() if provider else None
    ready = bool(caps and caps.generation_ready and selected_profile and script.strip())

    if caps:
        st.caption(f"Engine status: {caps.notes}")

    if st.button("Generate Speech", type="primary", disabled=not ready):
        result = provider.generate(script, language, dict(selected_profile))
        if result.get("status") == "completed":
            st.success("Generation complete.")
            gid = db.add_generation(script, language, caps.provider_id,
                                    selected_profile["id"], status="completed")
            audio_file_id = db.add_audio_file(
                result["audio_path"], "wav", 0
            )
            db.update_generation(gid, audio_file_id=audio_file_id)
        else:
            st.error(result.get("error", "Generation failed."))
            db.add_generation(script, language, caps.provider_id,
                              selected_profile["id"], status="failed",
                              error_message=result.get("error"))

    if not ready:
        st.info(
            "🔒 Generation is locked in Stage 1: no voice engine is connected yet. "
            "Configure a provider in Settings, create a voice profile, and connect "
            "a real engine in a later stage."
        )

    st.markdown("---")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Preview")
        st.caption("Audio preview appears here after a successful generation.")
    with right:
        st.markdown("#### Export")
        export_fmt = st.selectbox(
            "Export format", ["WAV", "MP3"],
            index=0 if config.get_setting("export_format", "wav") == "wav" else 1,
        )
        st.button("Export audio", disabled=True)
        st.caption("Export becomes available after generation is implemented.")
