"""Generation History page (Stage 3A-2): metadata, playback, download."""

from datetime import datetime

import streamlit as st

from .. import db, generation_service


def render() -> None:
    st.markdown('<div class="brand-title">Generation History</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">Every generation request, successful or not.'
                "</div>", unsafe_allow_html=True)

    rows = db.list_generations()
    if not rows:
        st.info("🔇 Nothing here yet. No generations have been recorded.")
        return

    st.dataframe(
        [
            {
                "Script": (r["script_text"][:80] + ("…" if len(r["script_text"]) > 80 else "")),
                "Language": r["language"],
                "Provider": r["provider"],
                "Model": r["provider_model"] or "—",
                "Voice profile": r["voice_profile_name"] or "—",
                "Emotion": r["emotion"] or "—",
                "Expression ref": r["expression_reference_name"] or "—",
                "Status": r["status"],
                "Created": r["created_at"][:19],
                "Has audio": bool(r["audio_path"]),
            }
            for r in rows
        ],
    )

    for r in rows:
        ts = datetime.fromisoformat(r["created_at"]).strftime("%Y-%m-%d %H:%M")
        with st.expander(f"{ts} · {r['provider']} · {r['status']}"):
            st.write(r["script_text"])
            st.caption(
                f"Voice profile: {r['voice_profile_name'] or '—'} · "
                f"Language: {r['language']} · Model: {r['provider_model'] or '—'}"
            )
            if r["emotion"]:
                st.caption(f"Emotion: {r['emotion']}")
            if r["expression_reference_name"]:
                st.caption(f"Expressive reference: {r['expression_reference_name']}")
            if r["error_message"]:
                st.error(r["error_message"])

            # Playback / download only for completed generations whose audio
            # still exists INSIDE the generated-audio directory. Records with
            # missing, deleted, or unsafe paths degrade to a caption.
            audio_path = generation_service.resolve_safe_audio_path(
                r["audio_path"])
            if audio_path:
                st.audio(str(audio_path), format="audio/mp3")
                st.download_button(
                    "Download MP3",
                    data=audio_path.read_bytes(),
                    file_name=audio_path.name,
                    mime="audio/mpeg",
                    key=f"hist_dl_{r['id']}",
                )
            elif r["status"] == "completed":
                st.caption("Audio file is no longer available on disk.")
