"""Voice Library page — manage voice profiles and their recordings."""

import streamlit as st

from .. import db, storage


NOTICE = (
    "🔒 **Privacy notice:** all voice recordings are stored **locally on this "
    "machine** only. They are never uploaded or transmitted anywhere unless you "
    "explicitly submit them to a cloud provider in a later stage."
)


def _voice_class_label(voice: dict) -> str:
    """Classify an ElevenLabs voice using its API `category` field.

    The ElevenLabs voice object carries both `category` and `is_owner`. They
    are NOT interchangeable:

      - `is_owner` is True only when the voice belongs to the calling account.
        It says nothing about HOW the voice was created.
      - `category` describes the voice's origin: 'premade', 'cloned',
        'generated', 'professional', or another value the API returns.

    We classify on `category` because that is the field that actually answers
    "is this a cloned voice or a stock voice". Unknown categories fall through
    to their raw value so the UI never silently mislabels a voice.
    """
    category = (voice.get("category") or "").strip().lower()
    mapping = {
        "cloned": "Cloned",
        "premade": "Premade",
        "generated": "Generated",
        "professional": "Professional",
    }
    return mapping.get(category, category.capitalize() if category else "Unknown")


def _render_sample(sample) -> None:
    col1, col2, col3 = st.columns([3, 2, 1])
    with col1:
        st.markdown(f"**{sample['original_filename']}**")
        st.caption(
            f"{sample['format'].upper()} · {sample['size_bytes'] / 1024:.0f} KB · "
            f"source: {sample['source']} · {sample['created_at'][:19]}"
        )
        st.caption(f"stored as `{sample['stored_filename']}`")
    with col2:
        st.audio(sample["file_path"])
    with col3:
        if st.button("🗑 Delete", key=f"del_{sample['id']}"):
            db.delete_voice_sample(sample["id"])
            st.rerun()
    # Reference transcript editor. The local cloning engine conditions on
    # this text, so it must be the user's own verified transcription — the
    # app never generates or guesses it. Empty means "not usable yet".
    has_transcript = bool((sample["transcript"] or "").strip())
    with st.expander(
            "📝 Reference transcript "
            + ("✅ set — ready for cloning" if has_transcript
               else "⚠️ missing — required for voice cloning"),
            expanded=not has_transcript):
        st.caption(
            "Type the exact words spoken in this recording. The local "
            "engine reads this text to learn which sounds belong to which "
            "letters — it cannot clone a voice without it.")
        new_transcript = st.text_area(
            "Transcript", value=sample["transcript"] or "",
            height=90, key=f"tr_{sample['id']}")
        col_save, col_clear = st.columns(2)
        if col_save.button("Save transcript", key=f"tr_save_{sample['id']}"):
            if not new_transcript.strip():
                st.error("The transcript is empty. Type what is said in the "
                         "recording, or clear it to mark the sample as "
                         "unusable.")
            else:
                db.update_sample_transcript(sample["id"],
                                            new_transcript.strip())
                st.success("Transcript saved. This sample is now usable for "
                           "voice cloning.")
                st.rerun()
        if col_clear.button("Clear transcript", key=f"tr_clear_{sample['id']}"):
            db.update_sample_transcript(sample["id"], None)
            st.info("Transcript cleared. This sample can no longer be used "
                    "for cloning until you enter it again.")
            st.rerun()


def render() -> None:
    st.markdown('<div class="brand-title">Voice Library</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-sub">Create voice profiles and manage the recordings '
        "that define them.</div>",
        unsafe_allow_html=True,
    )
    st.warning(NOTICE)

    tab_new, tab_manage = st.tabs(["➕ New profile", "📁 Manage profiles"])

    with tab_new:
        with st.form("new_profile"):
            name = st.text_input("Profile name")
            description = st.text_area("Description", height=80)
            language = st.selectbox("Preferred language", ["English", "Arabic"])
            if st.form_submit_button("Create profile", type="primary"):
                if not name.strip():
                    st.error("Profile name is required.")
                else:
                    try:
                        db.create_voice_profile(name, description, language)
                        st.success(f"Profile “{name}” created.")
                    except Exception as exc:  # e.g. duplicate name (UNIQUE)
                        st.error(f"Could not create profile: {exc}")

    with tab_manage:
        profiles = db.list_voice_profiles()
        if not profiles:
            st.info("No voice profiles yet — create one above.")
            return

        for profile in profiles:
            with st.expander(
                f"🎙 {profile['name']}  ·  {profile['preferred_language']}",
                expanded=False,
            ):
                st.caption(profile["description"] or "_No description_")

                with st.form(f"edit_{profile['id']}"):
                    new_name = st.text_input("Name", value=profile["name"], key=f"n_{profile['id']}")
                    new_desc = st.text_area("Description", value=profile["description"],
                                            height=70, key=f"d_{profile['id']}")
                    new_lang = st.selectbox("Preferred language",
                                            ["English", "Arabic"],
                                            index=["English", "Arabic"].index(profile["preferred_language"]),
                                            key=f"l_{profile['id']}")
                    if st.form_submit_button("Save changes"):
                        try:
                            db.update_voice_profile(profile["id"], new_name, new_desc, new_lang)
                            st.success("Profile updated.")
                        except Exception as exc:
                            st.error(f"Update failed: {exc}")

                up_col, rec_col = st.columns(2)

                with up_col:
                    uploads = st.file_uploader(
                        "Upload recordings",
                        type=["wav", "mp3", "m4a", "mp4", "aac", "flac",
                              "ogg", "opus", "webm", "mov", "mkv"],
                        accept_multiple_files=True,
                        key=f"up_{profile['id']}",
                    )
                    if uploads and st.button("Save uploads", key=f"save_{profile['id']}"):
                        for uf in uploads:
                            try:
                                meta = storage.store_upload(profile["id"], uf.name,
                                                            uf.getvalue(), source="upload")
                                db.add_voice_sample(
                                    profile["id"], **storage.sample_db_fields(meta))
                                st.success(f"Saved {uf.name}")
                            except storage.AudioValidationError as exc:
                                st.error(f"{uf.name}: {exc}")
                        st.rerun()

                with rec_col:
                    st.markdown("**Record with microphone**")
                    with st.form(f"rec_form_{profile['id']}", clear_on_submit=True):
                        audio_bytes = st.audio_input("Record a sample",
                                                     key=f"rec_{profile['id']}")
                        submitted = st.form_submit_button("Save recording")
                    if submitted:
                        if audio_bytes is None:
                            st.error("No recording captured. Please record before saving.")
                        else:
                            try:
                                meta = storage.store_upload(
                                    profile["id"], "recording.wav",
                                    audio_bytes.getvalue(), source="recording",
                                )
                                db.add_voice_sample(
                                    profile["id"], **storage.sample_db_fields(meta))
                                st.success("Recording saved.")
                                st.rerun()
                            except storage.AudioValidationError as exc:
                                st.error(f"Recording rejected: {exc}")

                samples = db.list_voice_samples(profile["id"])
                st.markdown(f"**Samples ({len(samples)})**")
                if not samples:
                    st.caption("No recordings yet for this profile.")
                for s in samples:
                    _render_sample(s)

                st.markdown("---")
                st.markdown("**ElevenLabs voice mapping**")
                mapping_key = f"el_map_{profile['id']}"
                current_id = profile["elevenlabs_voice_id"]
                if current_id:
                    st.caption(f"Mapped voice ID: `{current_id}`")
                else:
                    st.caption("No ElevenLabs voice mapped yet.")

                if st.button("Refresh ElevenLabs voices",
                             key=f"refresh_el_{profile['id']}"):
                    try:
                        from ..providers.elevenlabs import ElevenLabsProvider
                        voices = ElevenLabsProvider().list_voices()
                        st.session_state[f"el_voices_{profile['id']}"] = voices
                        st.session_state[mapping_key] = "refreshed"
                    except Exception as exc:
                        st.error(f"Could not fetch ElevenLabs voices: {exc}")

                    voices_cache = st.session_state.get(f"el_voices_{profile['id']}")
                    if voices_cache:
                        options = {"— unmapped —": None}
                        for v in voices_cache:
                            options[
                                f"{v['name']}  ·  {_voice_class_label(v)}"
                            ] = v["voice_id"]
                        # If the currently mapped voice ID is not present in the
                        # latest API response, keep it selectable rather than
                        # silently dropping the mapping to "— unmapped —".
                        if current_id and current_id not in options.values():
                            options[
                                f"Currently mapped: {current_id} "
                                f"(not in current voice list)"
                            ] = current_id
                        current_label = next(
                            (k for k, val in options.items() if val == current_id),
                            "— unmapped —",
                        )
                        chosen = st.selectbox(
                            "Mapped ElevenLabs voice",
                            list(options.keys()),
                            index=list(options.keys()).index(current_label),
                            key=f"el_choice_{profile['id']}",
                        )
                        chosen_id = options[chosen]
                        if st.button("Save mapping", key=f"save_el_{profile['id']}"):
                            try:
                                db.update_voice_profile_elevenlabs_id(
                                    profile["id"], chosen_id)
                                st.success("Mapping saved.")
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Could not save mapping: {exc}")
                        if current_id and st.button("Remove mapping",
                                                     key=f"rm_el_{profile['id']}"):
                            try:
                                db.update_voice_profile_elevenlabs_id(
                                    profile["id"], None)
                                st.success("Mapping removed.")
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Could not remove mapping: {exc}")
                    elif st.session_state.get(mapping_key) == "refreshed":
                        # The API returned an empty list, but a mapping may still
                        # exist from an earlier session — surface that fact.
                        if current_id:
                            st.info(
                                "No ElevenLabs voices are currently available, "
                                f"but this profile is still mapped to `{current_id}`."
                            )
                        else:
                            st.info("No ElevenLabs voices available for this account.")

                st.markdown("---")
                confirm_key = f"confirm_del_{profile['id']}"
                if st.button(f"Delete profile “{profile['name']}”", key=f"delp_{profile['id']}"):
                    st.session_state[confirm_key] = True
                if st.session_state.get(confirm_key):
                    if st.button("⚠️ Confirm: permanently delete profile and samples",
                                 key=f"delp2_{profile['id']}"):
                        for s in db.list_voice_samples(profile["id"]):
                            db.delete_voice_sample(s["id"])
                        db.delete_voice_profile(profile["id"])
                        st.session_state[confirm_key] = False
                        st.rerun()
