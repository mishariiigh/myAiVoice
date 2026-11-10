"""Generation History page."""

from datetime import datetime

import streamlit as st

from .. import db


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
                "Voice profile": r["voice_profile_name"] or "—",
                "Emotion": r["emotion"] or "—",
                "Expression ref": r["expression_reference_name"] or "—",
                "Status": r["status"],
                "Created": r["created_at"][:19],
                "Output file": r["audio_path"] or "—",
            }
            for r in rows
        ],
        use_container_width=True,
    )

    for r in rows:
        ts = datetime.fromisoformat(r["created_at"]).strftime("%Y-%m-%d %H:%M")
        with st.expander(f"{ts} · {r['provider']} · {r['status']}"):
            st.write(r["script_text"])
            if r["emotion"]:
                st.caption(f"Emotion: {r['emotion']}")
            if r["expression_reference_name"]:
                st.caption(f"Expressive reference: {r['expression_reference_name']}")
            if r["error_message"]:
                st.error(r["error_message"])
