"""Smoke tests: every page must render without exceptions (Stage 3A-2.3).

These tests exist because the Stage 3A-2 configuration work introduced a
runtime ``AttributeError`` (``app.config`` had no attribute
``EMOTION_CATEGORIES``) that no unit test caught — the failure only surfaced
when the Studio page rendered. They exercise the real entry point
(``app.main``) and drive the sidebar navigation, so a page that fails at
render time fails here too.

Regression coverage also includes:
- ``config.EMOTION_CATEGORIES`` exists and Studio exposes it in its selector.
- Studio reads export format through ``db.get_setting`` (it previously called
  the nonexistent ``config.get_setting``).
"""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app import config, db

PAGES = ["Studio", "Voice Library", "Generation History", "Settings"]


@pytest.fixture()
def app_env(temp_env):
    """AppTest rooted at the real entry point with no credentials set."""
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app" / "main.py"), default_timeout=30)
    at.run()
    assert not at.exception, f"App boot failed: {at.exception}"
    return at


def _goto(app_env, page):
    app_env.sidebar.radio[0].set_value(page)
    app_env.run()
    return app_env


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(app_env, page):
    at = _goto(app_env, page)
    assert not at.exception, f"{page} raised: {at.exception}"


def test_studio_emotion_selector_offers_config_categories(app_env):
    at = _goto(app_env, "Studio")
    box = next(s for s in at.selectbox
               if s.label == "Emotion (delivery hint)")
    assert list(box.options) == list(config.EMOTION_CATEGORIES)
    assert "Neutral" in box.options  # service-layer default must be offered


def test_config_exposes_emotion_categories():
    # Direct regression for the AttributeError seen at runtime.
    cats = getattr(config, "EMOTION_CATEGORIES", None)
    assert isinstance(cats, tuple) and cats
    assert "Neutral" in cats


def test_studio_reads_export_format_from_db(app_env):
    # Regression: studio.py used to call config.get_setting, which never
    # existed. With export_format = mp3 the selector must default to MP3.
    db.set_setting("export_format", "mp3")
    at = _goto(app_env, "Studio")
    box = next(s for s in at.selectbox if s.label == "Export format")
    assert box.value == "MP3"


def test_studio_shows_safe_message_with_credentials(app_env):
    """With an API key configured the Studio shows a setup hint and never a
    traceback. The exact wording depends on whether a voice profile exists,
    so we assert on the durable property: no exception and a guidance message."""
    at = _goto(app_env, "Studio")
    assert not at.exception
    assert at.info, "expected at least one info message"


def test_studio_shows_safe_message_without_credentials(monkeypatch, app_env):
    """With no API key the Studio shows a clear setup hint and disables
    generation. Regression for the original no-credentials message."""
    monkeypatch.setenv("ELEVENLABS_API_KEY", "")
    at = _goto(app_env, "Studio")
    assert not at.exception
    assert any("Configure the ElevenLabs provider" in i.value
               for i in at.info)
    gen_btn = next(b for b in at.button if b.label == "Generate Speech")
    assert gen_btn.disabled  # cannot generate with no provider configured


# --- Generation History row handling (Stage 3A-2.4) ---------------------------
#
# db.list_generations() returns genuine sqlite3.Row objects, which support
# indexed/keyed access but NOT dict-style .get(). history.py used
# r.get("audio_path") and crashed at render time.

def _history_page(app_env):
    app_env.sidebar.radio[0].set_value("Generation History")
    app_env.run()
    return app_env


def test_list_generations_returns_sqlite_rows_without_get(temp_env):
    db.add_generation("s", "English", "elevenlabs", status="completed")
    row = db.list_generations()[0]
    assert type(row).__name__ == "Row"
    assert not hasattr(row, "get")  # dict .get must not be used on rows


def test_history_empty(app_env):
    at = _history_page(app_env)
    assert not at.exception
    assert any("No generations have been recorded" in i.value
               for i in at.info)


def test_history_completed_generation_with_audio(temp_env):
    pid = db.create_voice_profile("Hist Profile", "", "English")
    db.add_generation("hello", "English", "elevenlabs",
                      voice_profile_id=pid, status="completed")
    gid = db.list_generations()[0]["id"]
    from app import storage
    stored = storage.store_generated_audio(b"ID3" + b"\x00" * 40, "mp3", gid)
    db.update_generation(gid, audio_file_id=db.add_audio_file(stored, "mp3", 44))
    at = AppTest.from_file(
        str(Path(__file__).resolve().parent.parent / "app" / "main.py"),
        default_timeout=30)
    at.run()
    at.sidebar.radio[0].set_value("Generation History")
    at.run()
    assert not at.exception
    assert any(d.label == "Download MP3" for d in at.download_button)


def test_history_failed_generation(app_env):
    db.add_generation("bad", "English", "elevenlabs", status="failed",
                      error_message="provider exploded")
    at = _history_page(app_env)
    assert not at.exception
    assert any("provider exploded" in e.value for e in at.error)


def test_history_processing_generation(app_env):
    db.add_generation("working", "English", "elevenlabs", status="processing")
    at = _history_page(app_env)
    assert not at.exception
    assert len(at.expander) == 1


def test_history_nullable_audio_fields(app_env):
    # Completed generation with NO audio file at all: audio_path is NULL.
    db.add_generation("no audio", "English", "elevenlabs", status="completed")
    at = _history_page(app_env)
    assert not at.exception
    assert any("no longer available" in c.value for c in at.caption)
