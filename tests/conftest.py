"""Test fixtures: redirect all storage into a temporary directory."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from app import config, db


@pytest.fixture()
def temp_env(tmp_path, monkeypatch):
    """Fresh temp storage + clean provider env for each test.

    Attributes are patched on the (shared) config module object so every
    importer sees the temporary paths; reloading would break references.
    """
    data_dir = tmp_path / "data"
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "VOICES_DIR", data_dir / "voices")
    monkeypatch.setattr(config, "SAMPLES_DIR", data_dir / "samples")
    monkeypatch.setattr(config, "GENERATED_DIR", data_dir / "generated")
    monkeypatch.setattr(config, "DB_PATH", data_dir / "test.db")
    monkeypatch.setattr(config, "ELEVENLABS_API_KEY", "")
    monkeypatch.setattr(config, "LOCAL_VOICE_MODEL_DIR", None)
    config.ensure_directories()
    db.init_db()
    yield data_dir

@pytest.fixture()
def real_upload_source():
    """Path to the user's real MP4 recording used as integration fixture.

    Skipped cleanly when the file is not present (e.g. on other machines) so
    the suite never fails for lack of a personal recording.
    """
    path = Path.home() / "Downloads" / "Recording 1.mp4"
    if not path.is_file():
        pytest.skip("Integration fixture 'Recording 1.mp4' not available")
    return path
