"""Studio page — text-to-speech workflow (local Audar and cloud engines)."""
from pathlib import Path


import streamlit as st

from .. import config, db, generation_service
from ..providers import get_registry
from ..providers.base import ProviderError


def render() -> None:
    st.markdown('<div class="brand-title">Studio</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-sub">Compose a script, pick a voice profile, and '
        "generate speech in English or Arabic.</div>",
        unsafe_allow_html=True,
    )

    registry = get_registry()
    profiles = db.list_voice_profiles()

    col1, col2 = st.columns(2)
    language = col1.selectbox("Language", list(config.SUPPORTED_LANGUAGES))
    # Engine choice: local (Audar, fully offline) or cloud (ElevenLabs).
    # Both engines appear here; validation below is strictly engine-specific.
    engine_options = {}
    for p in registry.list_providers():
        engine_caps = p.get_capabilities()
        if engine_caps.supports_local or engine_caps.supports_cloud:
            engine_options[engine_caps.name] = p
    chosen_name = col2.selectbox("Engine", list(engine_options.keys()))
    provider = engine_options.get(chosen_name)

    caps = provider.get_capabilities() if provider else None
    if caps:
        st.caption(f"Engine status: {caps.notes}")

    profile_names = [p["name"] for p in profiles] or ["No voice profiles yet"]
    profile_name = st.selectbox("Voice profile", profile_names)
    selected_profile = next(
        (p for p in profiles if p["name"] == profile_name), None)

    # Per-engine readiness for the chosen profile. The two engines have
    # DIFFERENT requirements: local needs a reference sample + its exact
    # transcript, ElevenLabs needs a mapped voice ID. Neither requirement
    # is ever applied to the other engine.
    mapped_voice_id = ""
    reference_ready = False
    if selected_profile:
        is_local = bool(caps and caps.supports_local and not caps.supports_cloud)
        if is_local:
            samples = db.list_voice_samples(selected_profile["id"])
            has_audio = any(Path(s["file_path"]).is_file() for s in samples)
            usable = db.get_usable_reference_sample(selected_profile["id"])
            reference_ready = usable is not None
            if reference_ready:
                st.caption("✅ Local reference ready — this profile can be "
                           "cloned with the Audar engine. No ElevenLabs "
                           "mapping needed.")
            elif has_audio or samples:
                st.warning(
                    "Reference audio is saved, but its exact transcript is "
                    "missing. Open the Voice Library, expand the sample and "
                    "type what is said in the recording.")
            else:
                st.warning(
                    "No reference audio uploaded yet. Upload or record a "
                    "voice sample in the Voice Library first.")
        else:
            mapped_voice_id = (selected_profile["elevenlabs_voice_id"] or "").strip()
            if mapped_voice_id:
                st.caption("✅ This profile is mapped to an ElevenLabs voice.")
            else:
                st.warning(
                    "This profile has no ElevenLabs voice mapped yet. Map it "
                    "in the Voice Library before generating.")

    script = st.text_area(
        "Script",
        height=220,
        placeholder="Type or paste the text you want to speak…",
    )

    emotion = st.selectbox("Emotion (delivery hint)", list(config.EMOTION_CATEGORIES))
    st.caption(
        "The emotion is saved with the generation. ElevenLabs does not accept "
        "an audio reference for conditioning in this workflow, so the emotion "
        "is recorded as metadata rather than guaranteed in the audio."
    )

    refs = db.list_expression_references(language=language, usable_only=True)
    ref_options = {"None": None}
    ref_options.update({
        f"{r['name']} ({r['emotion']})": r["id"] for r in refs
    })
    ref_label = st.selectbox(
        "Expression reference (optional)", list(ref_options.keys()))
    expression_reference_id = ref_options[ref_label]

    # Legacy local-engine placeholder stays, unchanged in spirit.
    st.markdown("---")
    is_local_engine = bool(caps and caps.supports_local and not caps.supports_cloud)
    profile_ready = reference_ready if is_local_engine else bool(mapped_voice_id)
    ready = bool(
        caps and caps.generation_ready and selected_profile
        and script.strip() and profile_ready
    )
    if st.button("Generate Speech", type="primary", disabled=not ready):
        with st.spinner("Generating speech…"):
            try:
                row = generation_service.run_generation(
                    provider, script, language, dict(selected_profile),
                    emotion=emotion,
                    expression_reference_id=expression_reference_id,
                )
                st.session_state["last_generation"] = row
            except ProviderError as exc:
                st.error(generation_service.build_generation_error(exc))
            except Exception:
                st.error("Generation failed unexpectedly. Please try again.")

    last = st.session_state.get("last_generation")
    if isinstance(last, dict) and last.get("status") == "completed":
        st.success("Generation complete.")
        audio_path = generation_service.resolve_safe_audio_path(
            last.get("audio_path"))
        if audio_path:
            st.audio(str(audio_path), format="audio/mp3")
            st.download_button(
                "Download MP3",
                data=audio_path.read_bytes(),
                file_name=audio_path.name,
                mime="audio/mpeg",
                key=f"dl_{last.get('id', 'latest')}",
            )
        else:
            st.warning("Generation completed, but the audio file could not "
                       "be located for playback.")

    if not ready:
        if not (caps and caps.generation_ready):
            if is_local_engine:
                st.info(
                    "The local voice engine is not fully installed. Check "
                    "that .venv-local and the Audar GGUF model exist."
                )
            else:
                st.info(
                    "Configure the ElevenLabs provider (Settings) to enable "
                    "generation."
                )
        elif profiles and selected_profile and not profile_ready:
            if is_local_engine:
                st.info("Add a voice sample and its exact transcript in the "
                        "Voice Library to enable local generation.")
            else:
                st.info("Map this voice profile to an ElevenLabs voice in the "
                        "Voice Library to enable generation.")
        elif not profiles:
            st.info("Create a voice profile in the Voice Library first.")

    # Export section (existing placeholder retained).
    st.markdown("---")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Preview")
        st.caption("Audio preview appears here after a successful generation.")
    with right:
        st.markdown("#### Export")
        export_fmt = st.selectbox(
            "Export format", ["WAV", "MP3"],
            index=0 if db.get_setting("export_format", "wav") == "wav" else 1,
        )
        st.button("Export audio", disabled=True)
        st.caption("Export becomes available after generation is implemented.")
