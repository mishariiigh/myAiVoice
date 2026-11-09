"""Voice Library page — manage voice profiles and their recordings."""

import streamlit as st

from .. import db, storage


NOTICE = (
    "🔒 **Privacy notice:** all voice recordings are stored **locally on this "
    "machine** only. They are never uploaded or transmitted anywhere unless you "
    "explicitly submit them to a cloud provider in a later stage."
)


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
                        type=["wav", "mp3", "m4a"],
                        accept_multiple_files=True,
                        key=f"up_{profile['id']}",
                    )
                    if uploads and st.button("Save uploads", key=f"save_{profile['id']}"):
                        for uf in uploads:
                            try:
                                meta = storage.store_upload(profile["id"], uf.name,
                                                            uf.getvalue(), source="upload")
                                db.add_voice_sample(profile["id"], **meta)
                                st.success(f"Saved {uf.name}")
                            except storage.AudioValidationError as exc:
                                st.error(f"{uf.name}: {exc}")
                        st.rerun()

                with rec_col:
                    st.markdown("**Record with microphone**")
                    audio_bytes = st.audio_input("Record a sample", key=f"rec_{profile['id']}")
                    if audio_bytes is not None and st.button("Save recording",
                                                             key=f"saverec_{profile['id']}"):
                        try:
                            meta = storage.store_upload(
                                profile["id"], "recording.wav", audio_bytes.getvalue(),
                                source="recording",
                            )
                            db.add_voice_sample(profile["id"], **meta)
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
