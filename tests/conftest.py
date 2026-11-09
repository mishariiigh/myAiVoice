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
