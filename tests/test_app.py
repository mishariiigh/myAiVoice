"""Pytest coverage for Mishari Voice Studio Stage 1."""

import sqlite3
import sys
import wave
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import config, db, storage  # noqa: E402
from app.providers import get_registry  # noqa: E402
from app.providers.base import (  # noqa: E402
    NotConfiguredError,
    NotImplementedYetError,
    Provider,
    ProviderRegistry,
)
from app.providers.elevenlabs import ElevenLabsProvider  # noqa: E402
from app.providers.local import LocalVoiceProvider  # noqa: E402


def make_wav_bytes(duration_s: float = 0.2, rate: int = 8000) -> bytes:
    import io, math, struct

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        for i in range(int(rate * duration_s)):
            w.writeframes(struct.pack("<h", int(12000 * math.sin(i * 0.05))))
    return buf.getvalue()


# --- 1 & 2: Database initialization and persistence ------------------------------

def test_db_initialization_creates_tables(temp_env):
    db.init_db()
    with sqlite3.connect(config.DB_PATH) as conn:
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"voice_profiles", "voice_samples", "scripts", "generations",
            "audio_files", "settings"} <= names


def test_db_persists_across_restart(temp_env):
    db.init_db()
    pid = db.create_voice_profile("Mishari Main", "primary voice", "Arabic")
    db.set_setting("export_format", "wav")
    # Simulate an application restart: reconnect from scratch
    with sqlite3.connect(config.DB_PATH) as conn:
        count = conn.execute("SELECT COUNT(*) FROM voice_profiles").fetchone()[0]
    assert count == 1
    assert db.get_voice_profile(pid)["name"] == "Mishari Main"
    assert db.get_setting("export_format") == "wav"


# --- 3: Voice profile creation and editing ----------------------------------------

def test_voice_profile_create_edit_delete(temp_env):
    pid = db.create_voice_profile("Narrator", "", "English")
    prof = db.get_voice_profile(pid)
    assert prof["preferred_language"] == "English"
    db.update_voice_profile(pid, name="Narrator v2", preferred_language="Arabic")
    prof = db.get_voice_profile(pid)
    assert prof["name"] == "Narrator v2"
    assert prof["preferred_language"] == "Arabic"
    db.delete_voice_profile(pid)
    assert db.get_voice_profile(pid) is None


def test_duplicate_profile_name_rejected(temp_env):
    db.create_voice_profile("Same", "", "English")
    with pytest.raises(sqlite3.IntegrityError):
        db.create_voice_profile("Same", "", "Arabic")


def test_invalid_language_rejected(temp_env):
    with pytest.raises(sqlite3.IntegrityError):
        db.create_voice_profile("Bad", "", "French")


# --- 4 & 5: Voice sample upload and audio validation -------------------------------

def test_valid_wav_upload_accepted(temp_env):
    pid = db.create_voice_profile("Tester", "", "English")
    raw = make_wav_bytes()
    meta = storage.store_upload(pid, "my voice.wav", raw, source="upload")
    assert meta["format"] == "wav"
    assert meta["duration_seconds"] == pytest.approx(0.2, abs=0.05)
    assert meta["sample_rate"] == 8000
    sid = db.add_voice_sample(pid, **meta)
    samples = db.list_voice_samples(pid)
    assert len(samples) == 1 and samples[0]["id"] == sid
    assert Path(meta["file_path"]).exists()


def test_corrupt_and_invalid_audio_rejected(temp_env):
    with pytest.raises(storage.AudioValidationError):
        storage.validate_audio("fake.wav", b"this is not audio at all")
    # truncated/corrupt WAV: correct header, broken body
    raw = make_wav_bytes()
    corrupt = bytearray(raw)
    corrupt[44:48] = b"XXXX"  # clobber fmt chunk size/data
    with pytest.raises(storage.AudioValidationError):
        storage.validate_audio("corrupt.wav", bytes(corrupt))
    # empty file
    with pytest.raises(storage.AudioValidationError):
        storage.validate_audio("empty.wav", b"")
    # oversize file
    big = make_wav_bytes() + b"\x00" * (config.MAX_UPLOAD_MB * 1024 * 1024 + 1)
    with pytest.raises(storage.AudioValidationError):
        storage.validate_audio("big.wav", big)
    # extension mismatch (real WAV named .mp3)
    with pytest.raises(storage.AudioValidationError):
        storage.validate_audio("mislabeled.mp3", make_wav_bytes())
    # unsupported extension
    with pytest.raises(storage.AudioValidationError):
        storage.validate_audio("song.ogg", b"OggS-decoy-data-here")


def test_mp3_signature_detected(temp_env):
    # MP3 detection relies on magic bytes; ID3-tagged file
    safe, fmt, *_ = storage.validate_audio(
        "clip.mp3", b"ID3\x03\x00\x00\x00\x00\x00\x00" + b"\xff\xfb" + b"\x00" * 64
    )
    assert fmt == "mp3" and safe == "clip.mp3"


# --- 6: Voice sample deletion -------------------------------------------------------

def test_delete_sample_removes_file_and_row(temp_env):
    pid = db.create_voice_profile("Del", "", "English")
    meta = storage.store_upload(pid, "take.wav", make_wav_bytes())
    sid = db.add_voice_sample(pid, **meta)
    path = Path(meta["file_path"])
    assert path.exists()
    db.delete_voice_sample(sid)
    assert db.get_voice_sample(sid) is None
    assert not path.exists()


# --- 7: Provider registry ------------------------------------------------------------

class DummyProvider(Provider):
    provider_id = "dummy"

    def get_capabilities(self):
        return None

    def validate_voice(self, voice_profile):
        return True

    def generate(self, script, language, voice_profile):
        return {"status": "not_implemented"}


def test_provider_registry_register_get_duplicate():
    reg = ProviderRegistry()
    p = DummyProvider()
    reg.register(p)
    assert reg.get("dummy") is p
    assert "dummy" in reg.ids()
    with pytest.raises(ValueError):
        reg.register(DummyProvider())
    reg.unregister("dummy")
    assert reg.get("dummy") is None


def test_default_registry_has_stage_one_providers():
    reg = get_registry()
    assert {"elevenlabs", "local"} <= set(reg.ids())
    assert isinstance(reg.get("elevenlabs"), ElevenLabsProvider)
    assert isinstance(reg.get("local"), LocalVoiceProvider)


# --- 8 & 9: Provider configuration status and missing credentials --------------------

def test_elevenlabs_missing_credentials_status(temp_env, monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    caps = ElevenLabsProvider().get_capabilities()
    assert caps.configured is False
    assert caps.generation_ready is False
    assert "ELEVENLABS_API_KEY" in " ".join(
        ElevenLabsProvider().missing_configuration())


def test_elevenlabs_configured_but_not_generating(temp_env, monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key-123")
    caps = ElevenLabsProvider().get_capabilities()
    assert caps.configured is True
    assert caps.generation_ready is False
    with pytest.raises(NotImplementedYetError):
        ElevenLabsProvider().generate("hello", "English", {})


def test_local_provider_unconfigured(temp_env):
    caps = LocalVoiceProvider().get_capabilities()
    assert caps.configured is False
    with pytest.raises(NotConfiguredError):
        LocalVoiceProvider().generate("مرحبا", "Arabic", {})


def test_generate_without_configuration_raises(temp_env, monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with pytest.raises(NotConfiguredError):
        ElevenLabsProvider().generate("hello", "English", {})


# --- 10: Storage path security --------------------------------------------------------

def test_path_traversal_blocked_on_upload(temp_env):
    with pytest.raises(storage.AudioValidationError):
        storage.validate_audio("../../etc/passwd.wav", make_wav_bytes())
    with pytest.raises(storage.AudioValidationError):
        storage.validate_audio("../secret.mp3", b"ID3 junksuffix")


def test_stored_paths_stay_inside_data_dir(temp_env):
    pid = db.create_voice_profile("Sec", "", "English")
    meta = storage.store_upload(pid, "ok.wav", make_wav_bytes())
    assert str(config.DATA_DIR) in Path(meta["file_path"]).resolve().__str__()


def test_delete_outside_data_dir_refused(temp_env, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "evil.wav"
    outside.write_bytes(make_wav_bytes())
    with pytest.raises(storage.AudioValidationError):
        storage.delete_sample_file(str(outside))
    assert outside.exists()


def test_no_secrets_in_capabilities_or_errors(temp_env, monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-secret-do-not-print")
    p = ElevenLabsProvider()
    rendered = str(p.get_capabilities()) + str(p.generate("x", "English", {}))
    assert "sk-secret-do-not-print" not in rendered
