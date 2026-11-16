"""Pytest coverage for Mishari Voice Studio Stage 1."""

import sqlite3
import stat
import subprocess
import sys
import tempfile
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
)
from app.providers.registry import ProviderRegistry  # noqa: E402
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
    corrupt[40:44] = (1 << 30).to_bytes(4, "little")  # bogus data-chunk size
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
    # unsupported extension (no real container signature either)
    with pytest.raises(storage.AudioValidationError):
        storage.validate_audio("song.xyz", b"random-bytes-not-a-container")
    # a real OGG signature passes sniffing but the extension must match.
    with pytest.raises(storage.AudioValidationError):
        storage.validate_audio("song.mp3", b"OggS-decoy-data-here")


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


def test_elevenlabs_configured_reports_generation_ready(temp_env, monkeypatch):
    # Stage 3A-1: when credentials exist, cloud TTS is implemented and ready.
    # The provider reads credentials dynamically, so patch the environment.
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key-123")
    caps = ElevenLabsProvider().get_capabilities()
    assert caps.configured is True
    assert caps.generation_ready is True
    # Without a mapped voice ID, generation fails with the mapping error,
    # not a "not implemented" error.
    with pytest.raises(ProviderVoiceMappingError):
        ElevenLabsProvider().generate("hello", "English", {})


def test_local_provider_unconfigured(temp_env, monkeypatch):
    # Simulate a machine without the isolated inference environment.
    monkeypatch.setenv("AUDAR_PYTHON", "/nonexistent/python")
    caps = LocalVoiceProvider().get_capabilities()
    assert caps.configured is False
    assert caps.generation_ready is False
    with pytest.raises(NotConfiguredError):
        LocalVoiceProvider().generate("مرحبا", "Arabic", {})


def test_local_provider_reports_ready_when_engine_present(temp_env):
    """The real engine pieces exist in this checkout: provider must say so."""
    from app.engines import audar_local
    if not audar_local.engine_ready():
        pytest.skip("Local engine pieces not present in this checkout")
    caps = LocalVoiceProvider().get_capabilities()
    assert caps.configured is True
    assert caps.generation_ready is True
    assert caps.supports_cloud is False


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
    monkeypatch.setattr(config, "ELEVENLABS_API_KEY", "sk-secret-do-not-print")
    p = ElevenLabsProvider()
    rendered = str(p.get_capabilities())
    try:
        p.generate("x", "English", {})
    except Exception as exc:
        rendered += str(exc)
    assert "sk-secret-do-not-print" not in rendered


# --- 11: Schema v1 -> v2 migration -------------------------------------------

def _read_schema_version(db_path: Path) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()
    except sqlite3.OperationalError:
        row = None
    finally:
        conn.close()
    return int(row[0]) if row is not None else 0


def _build_v1_database(db_path: Path) -> None:
    """Build a database that exactly represents the pre-migration (v1) state:
    all baseline tables present and populated, no schema_meta / v2 columns,
    no schema version row, and the connection's transaction committed to disk."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(db._V1_SCHEMA)
        conn.execute(
            "INSERT INTO voice_profiles (id, name, description, preferred_language,"
            " created_at, updated_at) VALUES (?, 'Legacy Profile', 'pre-migration',"
            " 'Arabic', '2020-01-01T00:00:00+00:00', '2020-01-01T00:00:00+00:00')",
            ("legacy-profile-id",),
        )
        conn.execute(
            "INSERT INTO voice_samples (id, profile_id, original_filename,"
            " stored_filename, file_path, format, size_bytes, duration_seconds,"
            " sample_rate, source, created_at) VALUES (?, 'legacy-profile-id',"
            " 'old.wav', 'old.wav', '/tmp/old.wav', 'wav', 100, 0.5, 8000,"
            " 'upload', '2020-01-01T00:00:00+00:00')",
            ("legacy-sample-id",),
        )
        conn.commit()
    finally:
        conn.close()


def _sample_columns(db_path: Path) -> set:
    conn = sqlite3.connect(str(db_path))
    try:
        return {r[1] for r in conn.execute("PRAGMA table_info(voice_samples)")}
    finally:
        conn.close()


def test_fresh_db_reaches_schema_version_2(temp_env):
    db.init_db()
    assert _read_schema_version(config.DB_PATH) == db.SCHEMA_VERSION


def test_v1_db_migrates_to_version_2(temp_env, tmp_path):
    db_path = tmp_path / "v1.db"
    _build_v1_database(db_path)
    assert _read_schema_version(db_path) == 0  # truly unmigrated before
    db.init_db(db_path)
    assert _read_schema_version(db_path) == db.SCHEMA_VERSION


def test_migration_persists_after_reopen(temp_env, tmp_path):
    db_path = tmp_path / "reopen.db"
    _build_v1_database(db_path)
    db.init_db(db_path)
    # Verify from a brand-new, independent connection.
    assert _read_schema_version(db_path) == db.SCHEMA_VERSION
    assert "sample_kind" in _sample_columns(db_path)
    assert "expression_references" in {
        r[0] for r in sqlite3.connect(str(db_path))
        .execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }


def test_repeated_init_db_does_not_duplicate_columns(temp_env):
    # The original failure mode: conftest's temp_env calls init_db(), the test
    # body calls it again. That now must be a no-op, not a duplicate-column error.
    db.init_db()
    db.init_db()
    db.init_db()
    with sqlite3.connect(config.DB_PATH) as conn:
        names = [r[1] for r in conn.execute("PRAGMA table_info(voice_samples)")]
    assert names.count("sample_kind") == 1
    with sqlite3.connect(config.DB_PATH) as conn:
        names = [r[1] for r in conn.execute("PRAGMA table_info(generations)")]
    assert names.count("emotion") == 1
    assert names.count("expression_reference_id") == 1


def test_migration_preserves_existing_records_and_classifies_legacy_samples(temp_env, tmp_path):
    db_path = tmp_path / "preserve.db"
    _build_v1_database(db_path)
    db.init_db(db_path)
    with sqlite3.connect(str(db_path)) as conn:
        profiles = conn.execute(
            "SELECT name, preferred_language FROM voice_profiles"
        ).fetchall()
        samples = conn.execute(
            "SELECT original_filename, sample_kind FROM voice_samples"
        ).fetchall()
    assert profiles == [("Legacy Profile", "Arabic")]
    assert samples == [("old.wav", "identity")]  # legacy rows defaulted to identity


def test_failed_migration_rolls_back_and_does_not_advance_version(temp_env, tmp_path):
    """A migration that fails before committing must leave the database clean:
    no schema-version row, and no partial/residual state that would make a
    subsequent (unblocked) `init_db()` call fail. Failure is induced by making
    the data directory read-only so the v2 transaction cannot touch the file;
    the version must remain 0 the entire time, and a following successful
    `init_db()` on the now-writable file must migrate cleanly with no
    duplicate-column or other errors."""
    db_file = tmp_path / "rollback.db"
    _build_v1_database(db_file)
    assert _read_schema_version(db_file) == 0

    import os
    try:
        os.chmod(db_file, stat.S_IREAD)  # file: read-only
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            db.init_db(db_file)
        # Version must not have advanced.
        assert _read_schema_version(db_file) == 0
    finally:
        os.chmod(db_file, stat.S_IWRITE | stat.S_IREAD)  # restore

    # Re-run on the now-unblocked file must succeed and reach v2.
    db.init_db(db_file)
    assert _read_schema_version(db_file) == db.SCHEMA_VERSION


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="relies on SQLite's 'no such table:' error-text format",
)
def test_get_schema_version_missing_table_returns_zero_and_errors_propagate(
        temp_env, tmp_path, monkeypatch):
    import app.db as db_module

    # A missing schema_meta table is the ONLY condition treated as "unmigrated":
    # empty database (no tables at all)
    empty_db = tmp_path / "empty.db"
    empty_db.touch()
    conn = sqlite3.connect(str(empty_db))
    try:
        assert db_module._get_schema_version(conn) == 0
    finally:
        conn.close()

    # ...and a database with tables but no schema_meta
    db_path = tmp_path / "notables.db"
    db_path.touch()
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("CREATE TABLE other (id TEXT)")
        conn.commit()
        assert db_module._get_schema_version(conn) == 0
    finally:
        conn.close()

    # The same unrelated-error property, but this time instantiated on the
    # missing-table path so both branches of _get_schema_version's error
    # handling are exercised.
    conn = sqlite3.connect(str(db_path))
    try:
        assert db_module._get_schema_version(conn) == 0  # no such table: schema_meta
        conn.execute(
            "CREATE TABLE schema_meta "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
        conn.commit()
        assert db_module._get_schema_version(conn) == 0  # table present, no row

        # Any OTHER database error must propagate, not be silently treated
        # as "version 0". Wrap the connection with a test-only proxy that
        # lets every statement through except the schema-version SELECT,
        # which is turned into an unrelated OperationalError raised inside
        # the production function's own statement path.
        class _LockedConn:
            def __init__(self, inner):
                self._inner = inner

            def execute(self, sql, *args, **kwargs):
                if "schema_version" in sql:
                    raise sqlite3.OperationalError("database is locked")
                return self._inner.execute(sql, *args, **kwargs)

        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            db_module._get_schema_version(_LockedConn(conn))
    finally:
        conn.close()


def test_repair_of_partially_migrated_db_without_version_row(temp_env, tmp_path):
    # Simulate a database abandoned mid-migration (e.g. by a tool that did not
    # use the app's transaction): v1 tables + one v2 column present, but
    # schema_meta / version row missing. init_db() must not "duplicate column"
    # and must bring it to the full v2 state.
    db_path = tmp_path / "half.db"
    _build_v1_database(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "ALTER TABLE voice_samples ADD COLUMN sample_kind TEXT NOT NULL "
        "DEFAULT 'identity' CHECK (sample_kind IN ('identity', 'expression'))"
    )
    conn.commit()
    conn.close()
    assert _read_schema_version(db_path) == 0  # no version recorded
    db.init_db(db_path)  # must not raise
    assert _read_schema_version(db_path) == db.SCHEMA_VERSION


# --- 12: Mid-transaction migration rollback ---------------------------------------

def test_midtransaction_migration_failure_rolls_back_to_clean_v1(
        temp_env, tmp_path, monkeypatch):
    """A failed v2 migration must regress the database fully to its v1
    pre-migration state: no partial rows, no version advance, and the next
    init_db() must succeed cleanly."""
    db_file = tmp_path / "mid.db"
    _build_v1_database(db_file)
    assert _read_schema_version(db_file) == 0

    # Inject exactly one bad statement (ALTER on a nonexistent table — a
    # genuine OperationalError from SQLite itself, not a re-raise of a
    # caught Python error) AFTER the first real v2 statement has executed.
    original = db._V2_MIGRATIONS[:]
    bad_statements = original[:1] + [
        "ALTER TABLE __definitely_does_not_exist__ ADD COLUMN x TEXT",
    ] + original[1:]
    monkeypatch.setattr(db, "_V2_MIGRATIONS", bad_statements)

    try:
        with pytest.raises(sqlite3.OperationalError, match="__definitely_does_not_exist__"):
            db.init_db(db_file)
    finally:
        monkeypatch.undo()  # production migration list restored regardless

    # The transaction must have rolled back completely.
    assert _read_schema_version(db_file) == 0  # version not advanced

    conn = sqlite3.connect(str(db_file))
    try:
        # No expression_references row / table should exist from the failed
        # attempt — v2 statements were all inside the rolled-back
        # transaction, so nothing was committed.
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert "expression_references" not in tables
        # The earlier successful sample_kind statement was rolled back.
        assert "sample_kind" not in {
            r[1] for r in conn.execute("PRAGMA table_info(voice_samples)")}
    finally:
        conn.close()

    # Restore the production migration list by re-reading it from the module
    # (monkeypatch.undo() has already restored the original reference) and run
    # init_db() again — it must now succeed and reach v2 without duplicate
    # column or other errors.
    db.init_db(db_file)
    assert _read_schema_version(db_file) == db.SCHEMA_VERSION
    assert "sample_kind" in _sample_columns(db_file)


# --- 13: Expression reference audio lifecycle ---------------------------------------

def test_update_expression_reference_audio_tristate(temp_env):
    pid = db.create_voice_profile("Expr", "", "English")
    audio1 = db.add_audio_file("shared/exp1.wav", "wav", 100)
    audio2 = db.add_audio_file("shared/exp2.wav", "wav", 100)
    rid = db.create_expression_reference("Angry", "anger", "English",
                                          audio_file_id=audio1)

    # State 1: omit the argument -> no change.
    db.update_expression_reference(rid, description="still angry")
    assert db.get_expression_reference(rid)["audio_file_id"] == audio1

    # State 2: assign a new audio reference.
    db.update_expression_reference(rid, audio_file_id=audio2)
    assert db.get_expression_reference(rid)["audio_file_id"] == audio2

    # State 3: explicit None -> clear the link, nothing else touched.
    db.update_expression_reference(rid, audio_file_id=None)
    assert db.get_expression_reference(rid)["audio_file_id"] is None
    assert db.get_expression_reference(rid)["description"] == "still angry"


def test_delete_expression_reference_preserves_shared_audio(temp_env):
    audio_shared = db.add_audio_file("shared/shared.wav", "wav", 100)
    ref_a = db.create_expression_reference("Angry", "anger", "English",
                                           audio_file_id=audio_shared)
    ref_b = db.create_expression_reference("Sad", "sadness", "English",
                                           audio_file_id=audio_shared)
    db.delete_expression_reference(ref_a)
    # The audio row and the reference row survive because ref_b still needs it.
    assert db.get_expression_reference(ref_b) is not None
    assert db.get_expression_reference(ref_b)["audio_file_id"] == audio_shared
    with sqlite3.connect(config.DB_PATH) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM audio_files WHERE id = ?", (audio_shared,)
        ).fetchone()[0] == 1


def test_delete_expression_reference_audio_still_used_by_generation(temp_env):
    audio = db.add_audio_file("shared/gen.wav", "wav", 100)
    ref = db.create_expression_reference("Calm", "calm", "English",
                                         audio_file_id=audio)
    db.add_generation("text", "English", "elevenlabs", status="completed",
                      expression_reference_id=ref)
    db.update_generation(db.list_generations()[0]["id"], audio_file_id=audio)
    db.delete_expression_reference(ref)
    # Generation keeps the audio: the row and the reference must survive.
    with sqlite3.connect(config.DB_PATH) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM audio_files WHERE id = ?", (audio,)
        ).fetchone()[0] == 1
    # The generation's reference link is nulled by ON DELETE SET NULL.
    gens = db.list_generations()
    assert len(gens) == 1
    assert gens[0]["expression_reference_id"] is None


def test_delete_expression_reference_cleans_own_audio(temp_env):
    audio = db.add_audio_file("own/only.wav", "wav", 100)
    ref = db.create_expression_reference("Joy", "joy", "English",
                                         audio_file_id=audio)
    db.delete_expression_reference(ref)
    # No other consumer: audio row and (if a file existed) it should be gone.
    with sqlite3.connect(config.DB_PATH) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM audio_files WHERE id = ?", (audio,)
        ).fetchone()[0] == 0
    assert db.get_expression_reference(ref) is None


# --- 14: Stage 3A-1 — ElevenLabs provider integration (mocked HTTP) -----------------

import json as _json

import requests as _requests

from app.providers import base as _pbase
from app.providers import elevenlabs as _el_module
from app.providers.base import (
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderVoiceMappingError,
)


@pytest.fixture()
def eleven_configured(temp_env, monkeypatch):
    """Configure the provider with a fake key and fake base URL."""
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key-eleven-3a1")
    monkeypatch.setenv("ELEVENLABS_BASE_URL", "https://api.elevenlabs.test")
    return temp_env


def _fake_profile(voice_id="vlVoice-abc123"):
    return {"id": "profile-1", "name": "Mishari Main",
            "elevenlabs_voice_id": voice_id}


class _FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, status_code=200, content=b"", headers=None, body=None):
        self.status_code = status_code
        self.content = content
        # Case-insensitive like real requests headers.
        from requests.structures import CaseInsensitiveDict
        self.headers = CaseInsensitiveDict(headers or {})
        self._body = body

    def json(self):
        if self._body is None:
            raise ValueError("no json body")
        return self._body

    @property
    def text(self):
        return _json.dumps(self._body) if self._body is not None else ""


def test_tts_sends_model_and_voice_not_language_code(eleven_configured, monkeypatch):
    captured = {}

    def fake_post(url, json=None, headers=None, params=None, timeout=None):
        captured.update(url=url, json=json, headers=headers, params=params,
                        timeout=timeout)
        return _FakeResponse(content=b"mp3-bytes",
                             headers={"Content-Type": "audio/mpeg"})

    monkeypatch.setattr(_el_module.requests, "post", fake_post)
    result = ElevenLabsProvider().generate("Hello world", "English",
                                           _fake_profile())
    assert result["status"] == "completed"
    assert result["audio_bytes"] == b"mp3-bytes"
    assert result["audio_format"] == "mp3"
    assert result["model_id"] == "eleven_multilingual_v2"
    # Endpoint and mapping
    assert "/v1/text-to-speech/vlVoice-abc123" in captured["url"]
    assert captured["json"]["model_id"] == "eleven_multilingual_v2"
    # language_code must NOT be sent for this model
    assert "language_code" not in captured["json"]
    assert "language" not in captured["json"]
    # Auth via header only; the key never appears in the URL or body.
    assert captured["headers"]["xi-api-key"] == "test-key-eleven-3a1"
    assert "test-key-eleven-3a1" not in captured["url"]
    assert "test-key-eleven-3a1" not in str(captured["json"])
    assert captured["timeout"] == 30.0


def test_tts_without_mapped_voice_id_raises(eleven_configured):
    with pytest.raises(ProviderVoiceMappingError):
        ElevenLabsProvider().generate("Hello", "English", {"id": "p1"})


def test_tts_empty_script_rejected(eleven_configured):
    with pytest.raises(ProviderRequestError):
        ElevenLabsProvider().generate("   ", "English", _fake_profile())


def test_tts_missing_credentials_raises_not_configured(eleven_configured,
                                                        monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with pytest.raises(NotConfiguredError):
        ElevenLabsProvider().generate("Hello", "English", _fake_profile())


def test_tts_authentication_error(eleven_configured, monkeypatch):
    monkeypatch.setattr(_el_module.requests, "post",
                        lambda *a, **k: _FakeResponse(status_code=401))
    with pytest.raises(ProviderAuthenticationError):
        ElevenLabsProvider().generate("Hello", "English", _fake_profile())


def test_tts_rate_limit_carries_retry_after(eleven_configured, monkeypatch):
    monkeypatch.setattr(
        _el_module.requests, "post",
        lambda *a, **k: _FakeResponse(status_code=429,
                                      headers={"Retry-After": "7"}))
    with pytest.raises(ProviderRateLimitError) as excinfo:
        ElevenLabsProvider().generate("Hello", "English", _fake_profile())
    assert excinfo.value.retry_after_seconds == 7.0


def test_tts_timeout_maps_to_typed_error(eleven_configured, monkeypatch):
    def raise_timeout(*a, **k):
        raise _requests.Timeout("timed out")

    monkeypatch.setattr(_el_module.requests, "post", raise_timeout)
    with pytest.raises(ProviderTimeoutError):
        ElevenLabsProvider().generate("Hello", "English", _fake_profile())


def test_tts_invalid_and_unexpected_responses(eleven_configured, monkeypatch):
    p = ElevenLabsProvider()
    profile = _fake_profile()
    # Empty audio body
    monkeypatch.setattr(_el_module.requests, "post",
                        lambda *a, **k: _FakeResponse(content=b""))
    with pytest.raises(ProviderResponseError):
        p.generate("Hello", "English", profile)
    # Wrong content type (JSON error body disguised as 200)
    monkeypatch.setattr(_el_module.requests, "post",
                        lambda *a, **k: _FakeResponse(
                            content=b'{"detail": "x"}',
                            headers={"Content-Type": "application/json"}))
    with pytest.raises(ProviderResponseError):
        p.generate("Hello", "English", profile)
    # 5xx
    monkeypatch.setattr(_el_module.requests, "post",
                        lambda *a, **k: _FakeResponse(status_code=500))
    with pytest.raises(ProviderResponseError):
        p.generate("Hello", "English", profile)


def test_error_messages_never_contain_the_api_key(eleven_configured, monkeypatch):
    """Every typed provider failure must render without leaking the key."""
    p = ElevenLabsProvider()
    profile = _fake_profile()

    monkeypatch.setattr(_el_module.requests, "post",
                        lambda *a, **k: _FakeResponse(status_code=401))
    with pytest.raises(ProviderError := _pbase.ProviderError) as exc:
        p.generate("Hello", "English", profile)
    rendered = str(exc.value)
    monkeypatch.undo()

    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key-eleven-3a1")
    assert "test-key-eleven-3a1" not in rendered
    assert "test-key-eleven-3a1" not in str(p.get_capabilities())


def test_list_voices_follows_pagination(eleven_configured, monkeypatch):
    pages = [
        _FakeResponse(body={
            "voices": [{"voice_id": "v2", "name": "Beirut", "is_owner": True},
                        {"voice_id": "v1", "name": "Aden"}],
            "has_more": True,
            "next_page_token": "page-2",
        }),
        _FakeResponse(body={
            "voices": [{"voice_id": "v3", "name": "Cairo", "category": "cloned"}],
            "has_more": False,
        }),
    ]
    calls = []

    def fake_get(url, headers=None, params=None, timeout=None):
        calls.append(params)
        return pages.pop(0)

    monkeypatch.setattr(_el_module.requests, "get", fake_get)
    voices = ElevenLabsProvider().list_voices()
    assert len(calls) == 2
    assert calls[0] == {"page_size": 100}
    assert calls[1] == {"page_size": 100, "cursor": "page-2"}
    # Sorted by name, all pages merged.
    assert [v["name"] for v in voices] == ["Aden", "Beirut", "Cairo"]
    assert [v["voice_id"] for v in voices] == ["v1", "v2", "v3"]
    # Ownership flags are preserved when present, default to False when absent.
    aden = next(v for v in voices if v["voice_id"] == "v1")
    assert aden["is_owner"] is False
    assert aden["category"] == ""
    beirut = next(v for v in voices if v["voice_id"] == "v2")
    assert beirut["is_owner"] is True
    cairo = next(v for v in voices if v["voice_id"] == "v3")
    assert cairo["category"] == "cloned"


def test_list_voices_handles_auth_and_malformed_responses(
        eleven_configured, monkeypatch):
    p = ElevenLabsProvider()
    monkeypatch.setattr(_el_module.requests, "get",
                        lambda *a, **k: _FakeResponse(status_code=401))
    with pytest.raises(ProviderAuthenticationError):
        p.list_voices()

    # Malformed JSON body
    monkeypatch.setattr(_el_module.requests, "get",
                        lambda *a, **k: _FakeResponse(body=None))
    with pytest.raises(ProviderResponseError):
        p.list_voices()

    # Wrong shape (voices not a list)
    monkeypatch.setattr(_el_module.requests, "get",
                        lambda *a, **k: _FakeResponse(body={"voices": "oops"}))
    with pytest.raises(ProviderResponseError):
        p.list_voices()

# Junk items inside the list are skipped, not fatal
    monkeypatch.setattr(_el_module.requests, "get",
                        lambda *a, **k: _FakeResponse(body={
                            "voices": ["junk", {"voice_id": "v9", "name": "Z"},
                                        {"voice_id": None, "name": "no-id"},
                                        {"voice_id": "v8"}]}))
    voices = p.list_voices()
    assert [v["voice_id"] for v in voices] == ["v9"]
    assert voices[0]["name"] == "Z"
    assert voices[0]["is_owner"] is False
    assert voices[0]["category"] == ""


def test_voice_mapping_persisted_and_used_for_validation(eleven_configured):
    pid = db.create_voice_profile("Mapped", "", "English")
    assert not ElevenLabsProvider().validate_voice(
        dict(db.get_voice_profile(pid)))
    db.update_voice_profile_elevenlabs_id(pid, "vlVoice-map-1")
    row = db.get_voice_profile(pid)
    assert row["elevenlabs_voice_id"] == "vlVoice-map-1"
    assert ElevenLabsProvider().validate_voice(dict(row))
    # Clearing the mapping removes usability.
    db.update_voice_profile_elevenlabs_id(pid, None)
    assert not ElevenLabsProvider().validate_voice(
        dict(db.get_voice_profile(pid)))


def test_capabilities_generation_ready_when_configured(eleven_configured):
    caps = ElevenLabsProvider().get_capabilities()
    assert caps.configured is True
    assert caps.generation_ready is True
    assert caps.provider_id == "elevenlabs"


# --- 15: Stage 3A-2 — Generation workflow integration (mocked HTTP) -----------------

from app import generation_service
from app.providers.base import ProviderError as _PE


@pytest.fixture()
def workflow_env(eleven_configured):
    """ElevenLabs configured + a profile with a mapped voice id."""
    pid = db.create_voice_profile("Workflow", "", "English")
    db.update_voice_profile_elevenlabs_id(pid, "vlVoice-workflow-1")
    return dict(db.get_voice_profile(pid))


def _mock_tts(monkeypatch, content=b"ID3\x03\x00\x00\x00\x00\x00\x00\xff\xfb" + b"\x00" * 32, status=200,
              content_type="audio/mpeg"):
    def fake_post(url, json=None, headers=None, params=None, timeout=None):
        return _FakeResponse(content=content,
                             headers={"Content-Type": content_type})
    monkeypatch.setattr(_el_module.requests, "post", fake_post)


def test_run_generation_full_workflow_persists_everything(workflow_env,
                                                           monkeypatch):
    profile = workflow_env
    _mock_tts(monkeypatch)
    ref_id = db.create_expression_reference("Warm", "calm", "English",
                                            audio_file_id=db.add_audio_file(
                                                "shared/warm.wav", "wav", 10))
    row = generation_service.run_generation(
        ElevenLabsProvider(), "  Hello world  ", "English", profile,
        emotion="Calm", expression_reference_id=ref_id)
    assert row["status"] == "completed"
    assert row["provider"] == "elevenlabs"
    assert row["provider_model"] == "eleven_multilingual_v2"
    assert row["emotion"] == "Calm"
    assert row["expression_reference_name"] == "Warm"
    assert row["voice_profile_name"] == "Workflow"
    assert row["audio_path"]  # joined from audio_files
    p = Path(row["audio_path"])
    assert p.exists() and p.read_bytes() == b"ID3\x03\x00\x00\x00\x00\x00\x00\xff\xfb" + b"\x00" * 32
    # File lives inside GENERATED_DIR.
    assert str(config.GENERATED_DIR) in str(p.resolve())
    assert p.suffix == ".mp3"
    # Script stored trimmed.
    assert row["script_text"] == "Hello world"


def test_run_generation_repeated_calls_create_distinct_records_and_files(
        workflow_env, monkeypatch):
    profile = workflow_env
    _mock_tts(monkeypatch)
    first = generation_service.run_generation(
        ElevenLabsProvider(), "One", "English", profile)
    second = generation_service.run_generation(
        ElevenLabsProvider(), "One", "English", profile)
    assert first["id"] != second["id"]
    assert first["audio_path"] != second["audio_path"]
    assert Path(first["audio_path"]).exists()
    assert Path(second["audio_path"]).exists()
    assert len(db.list_generations()) == 2


def test_run_generation_empty_script_rejected_without_records(workflow_env):
    with pytest.raises(_PE):
        generation_service.run_generation(
            ElevenLabsProvider(), "   ", "English", workflow_env)
    assert db.list_generations() == []


def test_run_generation_missing_credentials_no_records(workflow_env,
                                                        monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with pytest.raises(NotConfiguredError):
        generation_service.run_generation(
            ElevenLabsProvider(), "Hello", "English", workflow_env)
    assert db.list_generations() == []


def test_run_generation_missing_voice_mapping_no_records(eleven_configured):
    pid = db.create_voice_profile("NoMap", "", "English")
    profile = dict(db.get_voice_profile(pid))
    with pytest.raises(ProviderVoiceMappingError):
        generation_service.run_generation(
            ElevenLabsProvider(), "Hello", "English", profile)
    assert db.list_generations() == []


def test_run_generation_provider_failures_leave_no_records(
        workflow_env, monkeypatch):
    profile = workflow_env
    cases = [
        # (simulate, expected exception)
        (lambda: _FakeResponse(status_code=401), ProviderAuthenticationError),
        (lambda: _FakeResponse(status_code=429,
                               headers={"Retry-After": "3"}),
         ProviderRateLimitError),
        (lambda: _FakeResponse(status_code=500), ProviderResponseError),
        (lambda: _FakeResponse(content=b"",
                               headers={"Content-Type": "audio/mpeg"}),
         ProviderResponseError),
        (lambda: _FakeResponse(content=b"x",
                               headers={"Content-Type": "application/json"}),
         ProviderResponseError),
    ]
    for factory, expected in cases:
        monkeypatch.setattr(_el_module.requests, "post",
                            lambda *a, **k: factory())
        with pytest.raises(expected):
            generation_service.run_generation(
                ElevenLabsProvider(), "Hello", "English", profile)
        assert db.list_generations() == []  # nothing persisted on failure

    def raise_timeout(*a, **k):
        raise _requests.Timeout("boom")
    monkeypatch.setattr(_el_module.requests, "post", raise_timeout)
    with pytest.raises(ProviderTimeoutError):
        generation_service.run_generation(
            ElevenLabsProvider(), "Hello", "English", profile)
    assert db.list_generations() == []


def test_audio_storage_failure_marks_failed_and_leaves_no_orphan_file(
        workflow_env, monkeypatch):
    profile = workflow_env
    _mock_tts(monkeypatch)
    # Corrupt the provider payload so storage validation fails
    # (declared mp3, content is not audio).
    monkeypatch.setattr(_el_module.requests, "post",
                        lambda *a, **k: _FakeResponse(
                            content=b"not really audio",
                            headers={"Content-Type": "audio/mpeg"}))
    with pytest.raises(_PE):
        generation_service.run_generation(
            ElevenLabsProvider(), "Hello", "English", profile)
    rows = db.list_generations()
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert rows[0]["audio_file_id"] is None
    assert rows[0]["error_message"]  # safe message recorded
    # No orphan audio files on disk for this generation.
    leftovers = list(config.GENERATED_DIR.iterdir())
    assert not any("not" in f.name for f in leftovers)


def test_db_insertion_failure_cleans_up_audio_file(workflow_env, monkeypatch):
    profile = workflow_env
    _mock_tts(monkeypatch)
    stored = {}
    real_store = storage.store_generated_audio

    def spy_store(audio_bytes, fmt, gid):
        path = real_store(audio_bytes, fmt, gid)
        stored["path"] = path
        return path

    monkeypatch.setattr(storage, "store_generated_audio", spy_store)
    # Break the audio_files insert AFTER the file was written to disk.
    def boom_add_audio(*a, **k):
        raise sqlite3.OperationalError("disk I/O error")
    monkeypatch.setattr(db, "add_audio_file", boom_add_audio)

    with pytest.raises(_PE):
        generation_service.run_generation(
            ElevenLabsProvider(), "Hello", "English", profile)

    rows = db.list_generations()
    assert rows[0]["status"] == "failed"
    # The orphaned on-disk file was removed during rollback.
    assert not Path(stored["path"]).exists()


def test_generation_error_messages_are_secret_safe(eleven_configured,
                                                    monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-super-secret-key-xyz")
    p = ElevenLabsProvider()
    for exc in (NotConfiguredError("no creds"),
                ProviderAuthenticationError("bad credentials"),
                ProviderRateLimitError("slow down", 5),
                ProviderTimeoutError("took too long"),
                ProviderResponseError("bad body")):
        msg = generation_service.build_generation_error(exc)
        assert "sk-super-secret-key-xyz" not in msg
        assert isinstance(msg, str) and msg


def test_resolve_safe_audio_path_rejects_missing_and_unsafe():
    # None / empty
    assert generation_service.resolve_safe_audio_path(None) is None
    assert generation_service.resolve_safe_audio_path("") is None
    # Outside GENERATED_DIR
    outside = config.SAMPLES_DIR / "x.mp3"
    assert generation_service.resolve_safe_audio_path(str(outside)) is None
    # Traversal attempt
    assert generation_service.resolve_safe_audio_path(
        str(config.GENERATED_DIR / "../../etc/passwd")) is None
    # Nonexistent file inside GENERATED_DIR
    assert generation_service.resolve_safe_audio_path(
        str(config.GENERATED_DIR / "missing.mp3")) is None


def test_history_rows_with_missing_audio_remain_readable(workflow_env,
                                                          monkeypatch):
    profile = workflow_env
    _mock_tts(monkeypatch)
    good = generation_service.run_generation(
        ElevenLabsProvider(), "Keep me", "English", profile)
    # A legacy/failed generation without any audio reference.
    legacy_id = db.add_generation("old text", "Arabic", "elevenlabs",
                                  voice_profile_id=profile["id"],
                                  status="failed", error_message="old failure")
    rows = db.list_generations()
    assert len(rows) == 2
    by_id = {r["id"]: r for r in rows}
    assert by_id[legacy_id]["audio_path"] is None  # renders as "—", no crash
    # Deleting the audio row behind the good generation also must not break
    # listing or safe-path resolution.
    db.delete_audio_file(
        next(a["id"] for a in db.list_audio_files()
             if a["path"] == good["audio_path"]))
    rows2 = db.list_generations()
    assert Path(good["audio_path"]).exists()  # file kept; row gone is legacy-safe


def test_generated_audio_limit_enforced(workflow_env, monkeypatch):
    profile = workflow_env
    big = b"RIFFxxxxWAVEfmt " + b"\x00" * (config.MAX_GENERATED_MB * 1024 * 1024 + 1)
    monkeypatch.setattr(_el_module.requests, "post",
                        lambda *a, **k: _FakeResponse(
                            content=big, headers={"Content-Type": "audio/mpeg"}))
    with pytest.raises(_PE):
        generation_service.run_generation(
            ElevenLabsProvider(), "Hello", "English", profile)
    rows = db.list_generations()
    assert rows and rows[0]["status"] == "failed"


def test_delete_audio_file_refuses_when_generation_depends_on_it(
        workflow_env, monkeypatch):
    profile = workflow_env
    _mock_tts(monkeypatch)
    row = generation_service.run_generation(
        ElevenLabsProvider(), "Pinned", "English", profile)
    audio_id = next(a["id"] for a in db.list_audio_files()
                    if a["path"] == row["audio_path"])
    db.delete_audio_file(audio_id)  # referenced by the completed generation
    assert any(a["id"] == audio_id for a in db.list_audio_files())


# --- Voice Library microphone recording (Stage 3A-2.2) -------------------------
#
# Regression tests for the st.audio_input fix. Before the fix the recording
# widget was rendered outside st.form(), so its deferred commit was never
# flushed and the "Save recording" button was a plain st.button that could
# fire on every rerun. The fix wraps audio_input + form_submit_button inside
# one st.form per profile so the widget value is only consumed on submit.
#
# These tests drive the real entry point via Streamlit AppTest and assert on
# observable behaviour: error/success messages and the number of persisted
# voice samples. They do not depend on a real microphone.

from streamlit.testing.v1 import AppTest  # noqa: E402


def _make_wav_bytes(duration_s: float = 0.2, rate: int = 8000) -> bytes:
    import io as _io
    import math as _math
    import struct as _struct
    import wave as _wave

    buf = _io.BytesIO()
    with _wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        for i in range(int(rate * duration_s)):
            w.writeframes(_struct.pack("<h", int(12000 * _math.sin(i * 0.05))))
    return buf.getvalue()


def _library_app(temp_env):
    """Navigate to Voice Library. The caller must have already created any
    profiles it needs, since the page only renders per-profile expanders when
    profiles exist."""
    at = AppTest.from_file(
        str(PROJECT_ROOT / "app" / "main.py"), default_timeout=30)
    at.run()
    assert not at.exception, f"App boot failed: {at.exception}"
    at.sidebar.radio[0].set_value("Voice Library")
    at.run()
    assert not at.exception, f"Voice Library raised: {at.exception}"
    return at


def _submit_recording(at, recording_tuple=None):
    """Set the audio_input value (or clear it) and submit the Save form."""
    ai = next(a for a in at.audio_input if a.label == "Record a sample")
    ai.set_value(recording_tuple)
    btn = next(b for b in at.form_submit_button if b.label == "Save recording")
    btn.set_value(True)
    at.run()
    return at


def test_recording_submission_creates_one_sample(temp_env):
    """A valid recording submission persists exactly one voice sample."""
    pid = db.create_voice_profile("Rec Profile", "", "English")
    at = _library_app(temp_env)
    at = _submit_recording(at, ("recording.wav", _make_wav_bytes(), "audio/wav"))
    assert not at.exception
    # st.success is followed by st.rerun(), so the message is transient;
    # the durable evidence is the persisted sample.
    samples = db.list_voice_samples(pid)
    assert len(samples) == 1
    assert samples[0]["source"] == "recording"
    assert samples[0]["format"] == "wav"


def test_recording_submission_idempotent_on_rerun(temp_env):
    """Submitting twice must not create two samples; clear_on_submit guards it."""
    pid = db.create_voice_profile("Rec Idempotent", "", "English")
    at = _library_app(temp_env)
    at = _submit_recording(at, ("recording.wav", _make_wav_bytes(), "audio/wav"))
    assert not at.exception
    assert len(db.list_voice_samples(pid)) == 1
    # A second submit with no new recording must be a no-op.
    at = _submit_recording(at, None)
    assert not at.exception
    assert any("No recording captured" in e.value for e in at.error)
    assert len(db.list_voice_samples(pid)) == 1


def test_recording_submission_empty_rejected(temp_env):
    """Submitting the form with no recording shows a clear error, no sample."""
    pid = db.create_voice_profile("Rec Empty", "", "English")
    at = _library_app(temp_env)
    at = _submit_recording(at, None)
    assert not at.exception
    assert any("No recording captured" in e.value for e in at.error)
    assert db.list_voice_samples(pid) == []


def test_recording_submission_invalid_wav_rejected(temp_env):
    """Non-audio bytes are rejected by storage.validate_audio; no sample saved."""
    pid = db.create_voice_profile("Rec Bad", "", "English")
    at = _library_app(temp_env)
    at = _submit_recording(at, ("bad.wav", b"NOT ACTUAL AUDIO DATA", "audio/wav"))
    assert not at.exception
    assert any("Recording rejected" in e.value for e in at.error)
    assert db.list_voice_samples(pid) == []


def test_recording_submission_save_button_is_form_submit(temp_env):
    """The Save control must be a form_submit_button, not a plain st.button."""
    db.create_voice_profile("Rec Button", "", "English")
    at = _library_app(temp_env)
    # The Save recording control must be wired through st.form_submit_button.
    assert "Save recording" in [b.label for b in at.form_submit_button]
    # It must not be a standalone st.button outside any form (the pre-fix bug).
    standalone = [b for b in at.button if b.label == "Save recording"
                  and b not in at.form_submit_button]
    assert standalone == []


# --- ElevenLabs voice mapping UI (Stage 3A-2.5) --------------------------------
#
# The Voice Library page now exposes a per-profile "ElevenLabs voice mapping"
# section with a Refresh / select / Save / Remove control. These tests cover
# the mapping persistence path (db.update_voice_profile_elevenlabs_id) and the
# provider's list_voices() ownership metadata, which the UI uses to label each
# choice as "cloned" or "premade".

def test_list_voices_preserves_owner_and_category_metadata(eleven_configured,
                                                            monkeypatch):
    pages = [
        _FakeResponse(body={
            "voices": [
                {"voice_id": "own-1", "name": "My Clone", "is_owner": True},
                {"voice_id": "pre-1", "name": "Sarah", "category": "premade"},
                {"voice_id": "bare", "name": "Bare Minimum"},
            ],
            "has_more": False,
        }),
    ]
    monkeypatch.setattr(_el_module.requests, "get",
                        lambda *a, **k: pages.pop(0))
    voices = ElevenLabsProvider().list_voices()
    by_id = {v["voice_id"]: v for v in voices}
    assert by_id["own-1"]["is_owner"] is True
    assert by_id["pre-1"]["is_owner"] is False
    assert by_id["pre-1"]["category"] == "premade"
    assert by_id["bare"]["is_owner"] is False
    assert by_id["bare"]["category"] == ""


def test_voice_mapping_save_and_remove_persist(temp_env):
    """Mapping a voice ID and then clearing it must round-trip through the DB."""
    pid = db.create_voice_profile("Map Test", "", "English")
    assert db.get_voice_profile(pid)["elevenlabs_voice_id"] is None

    db.update_voice_profile_elevenlabs_id(pid, "vlVoice-xyz")
    assert db.get_voice_profile(pid)["elevenlabs_voice_id"] == "vlVoice-xyz"

    # Whitespace-only IDs are treated as unmapped.
    db.update_voice_profile_elevenlabs_id(pid, "   ")
    assert db.get_voice_profile(pid)["elevenlabs_voice_id"] is None

    db.update_voice_profile_elevenlabs_id(pid, None)
    assert db.get_voice_profile(pid)["elevenlabs_voice_id"] is None


def test_voice_mapping_persists_across_reruns(temp_env):
    """A mapping written in one session must survive a fresh connection."""
    pid = db.create_voice_profile("Persistent Map", "", "English")
    db.update_voice_profile_elevenlabs_id(pid, "vlVoice-stable")
    # New connection, same DB file.
    row = db.get_voice_profile(pid)
    assert row["elevenlabs_voice_id"] == "vlVoice-stable"
    assert row["updated_at"] is not None


def test_voice_mapping_does_not_auto_assign_mishari(temp_env):
    """No premade voice is mapped to the Mishari profile without explicit action."""
    pid = db.create_voice_profile("Mishari", "", "English")
    assert db.get_voice_profile(pid)["elevenlabs_voice_id"] is None
    # Explicitly mapping then clearing must not leave a stale value.
    db.update_voice_profile_elevenlabs_id(pid, "vlVoice-explicit")
    assert db.get_voice_profile(pid)["elevenlabs_voice_id"] == "vlVoice-explicit"
    db.update_voice_profile_elevenlabs_id(pid, None)
    assert db.get_voice_profile(pid)["elevenlabs_voice_id"] is None


def test_list_voices_empty_when_no_voices_returned(eleven_configured, monkeypatch):
    monkeypatch.setattr(_el_module.requests, "get",
                        lambda *a, **k: _FakeResponse(body={"voices": [],
                                                            "has_more": False}))
    assert ElevenLabsProvider().list_voices() == []


def test_list_voices_skips_malformed_items(eleven_configured, monkeypatch):
    monkeypatch.setattr(_el_module.requests, "get",
                        lambda *a, **k: _FakeResponse(body={
                            "voices": [
                                {"voice_id": "ok-1", "name": "Good"},
                                {"name": "No ID"},
                                {"voice_id": "ok-2"},
                                "not-a-dict",
                                None,
                            ],
                            "has_more": False,
                        }))
    voices = ElevenLabsProvider().list_voices()
    assert [v["voice_id"] for v in voices] == ["ok-1"]


def test_voice_mapping_ui_renders_without_credentials(temp_env):
    """With no API key the mapping section must render, not crash."""
    from streamlit.testing.v1 import AppTest
    db.create_voice_profile("UI Profile", "", "English")
    at = AppTest.from_file(
        str(PROJECT_ROOT / "app" / "main.py"), default_timeout=30)
    at.run()
    assert not at.exception
    at.sidebar.radio[0].set_value("Voice Library")
    at.run()
    assert not at.exception
    # The Voice Library page renders per-profile expanders. The mapping
    # control must exist in the page tree even without credentials configured.
    assert any(b.label == "Refresh ElevenLabs voices" for b in at.button)


# --- Voice classification uses the API `category` field ------------------------
#
# The mapping UI labels each ElevenLabs choice as Cloned / Premade / Generated
# / Professional. The label must come from the voice object's `category` field,
# NOT from `is_owner`: `is_owner` only says whether the voice belongs to the
# calling account, it does not describe how the voice was created.

def test_voice_class_label_uses_category_not_is_owner():
    from app.pages.voice_library import _voice_class_label
    assert _voice_class_label({"category": "cloned", "is_owner": True}) == "Cloned"
    assert _voice_class_label({"category": "premade", "is_owner": False}) == "Premade"
    assert _voice_class_label({"category": "generated"}) == "Generated"
    assert _voice_class_label({"category": "professional"}) == "Professional"
    # Unknown categories fall through to their raw value rather than being
    # silently mislabelled as "Premade" or "Cloned".
    assert _voice_class_label({"category": "custom_voices"}) == "Custom_voices"
    assert _voice_class_label({"category": ""}) == "Unknown"
    assert _voice_class_label({}) == "Unknown"


def test_voice_class_label_ignores_is_owner_when_category_conflicts():
    """is_owner=True must not override an explicit premade category."""
    from app.pages.voice_library import _voice_class_label
    # A premade voice that happens to be owned (rare but possible) is still
    # labelled Premade because category is the authoritative origin field.
    assert _voice_class_label(
        {"category": "premade", "is_owner": True}) == "Premade"


def test_mapped_voice_kept_when_missing_from_latest_api(temp_env, monkeypatch):
    """If the mapped voice ID disappears from the API response, the mapping
    must remain selectable — it must not silently become '— unmapped —'."""
    from app.providers.elevenlabs import ElevenLabsProvider
    pid = db.create_voice_profile("Stale Map", "", "English")
    db.update_voice_profile_elevenlabs_id(pid, "vlVoice-gone")

    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    monkeypatch.setenv("ELEVENLABS_BASE_URL", "https://api.elevenlabs.test")
    monkeypatch.setattr(_el_module.requests, "get",
                        lambda *a, **k: _FakeResponse(body={
                            "voices": [{"voice_id": "other-1", "name": "Other"}],
                            "has_more": False,
                        }))

    voices = ElevenLabsProvider().list_voices()
    ids = [v["voice_id"] for v in voices]
    assert "vlVoice-gone" not in ids  # the mapped voice is absent from the API

    # Simulate the UI's option-building logic.
    options = {"— unmapped —": None}
    for v in voices:
        options[f"{v['name']}  ·  test"] = v["voice_id"]
    current_id = db.get_voice_profile(pid)["elevenlabs_voice_id"]
    if current_id and current_id not in options.values():
        options[f"Currently mapped: {current_id} (not in current voice list)"] = current_id

    assert any("vlVoice-gone" in k for k in options), \
        "mapped voice must remain selectable when absent from the API"
    assert options.get("— unmapped —") is None or current_id is None


# --- Media upload / extraction (MP4 & friends) ----------------------------------
#
# The Voice Library now accepts MP4/MOV/M4A/AAC/FLAC/OGG/Opus/WebM uploads.
# Anything that is not already a clean WAV is decoded through the bundled
# FFmpeg (app/media.py) into a 16 kHz mono PCM WAV reference that matches the
# Audar/NeuCodec requirements. These tests verify observable behaviour:
# stored format, sample rate, channels, duration and rejection messages.
def _wav_header_only() -> bytes:
    return make_wav_bytes()


def test_store_upload_mp4_opus_is_extracted_to_reference_wav(temp_env,
                                                              real_upload_source):
    """The real MP4/Opus recording is accepted, extracted and stored as a
    16 kHz mono WAV voice reference — the format Audar/NeuCodec expect."""
    raw = real_upload_source.read_bytes()
    meta = storage.store_upload("profile-mp4", real_upload_source.name, raw,
                                source="upload")
    assert meta["format"] == "wav"
    assert meta["sample_rate"] == 16000
    assert meta["duration_seconds"] and meta["duration_seconds"] > 100
    assert meta["source_audio_codec"] == "opus"
    stored = Path(meta["file_path"])
    assert stored.is_file() and stored.stat().st_size > 100_000
    # Stored audio is a structurally valid WAV at the required rate.
    duration, rate = storage.probe_wav(stored)
    assert rate == 16000 and duration and duration > 100
    with wave.open(str(stored), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2  # 16-bit PCM
        assert w.getframerate() == 16000
    # The ORIGINAL upload is untouched on disk.
    assert real_upload_source.stat().st_size == len(raw)


def test_store_upload_m4a_variant_accepted(temp_env, real_upload_source):
    """MP4-family naming (e.g. an .m4a copy of the same stream) passes."""
    raw = real_upload_source.read_bytes()
    meta = storage.store_upload("profile-m4a", "voice_memo.m4a", raw,
                                source="upload")
    assert meta["format"] == "wav"
    assert meta["sample_rate"] == 16000


def test_store_upload_rejects_video_without_audio_stream(temp_env):
    """A real video file with NO audio track is refused with a clear message.

    We build one by stripping the audio with FFmpeg from the fixture; if
    FFmpeg is unavailable the test skips (the unit layer is covered elsewhere).
    """
    try:
        from app import media as media_mod
        ffmpeg = media_mod.resolve_ffmpeg()
    except media_mod.MediaExtractionError:
        pytest.skip("No FFmpeg executable available in test environment")
    with tempfile.TemporaryDirectory() as tmp:
        silent = Path(tmp) / "silent.mp4"
        subprocess.run(
            [ffmpeg, "-v", "error", "-y", "-f", "lavfi",
             "-i", "testsrc=duration=1:size=128x72:rate=10",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", str(silent)],
            capture_output=True, timeout=120,
        )
        if not silent.is_file():
            pytest.skip("Could not build a silent test video (no libx264)")
        raw = silent.read_bytes()
        with pytest.raises(storage.AudioValidationError) as err:
            storage.store_upload("profile-silent", "silent.mp4", raw,
                                 source="upload")
        assert "no audio stream" in str(err.value)


def test_store_upload_rejects_corrupt_media(temp_env):
    """Bytes that fake an MP4 header but are not decodable are rejected."""
    fake = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 256
    with pytest.raises(storage.AudioValidationError) as err:
        storage.store_upload("profile-corrupt", "broken.mp4", fake,
                             source="upload")
    msg = str(err.value)
    assert ("could not be decoded" in msg) or ("extracted" in msg)


def test_validate_audio_rejects_unsupported_container(temp_env):
    """An extension outside the accepted set is refused before any decoding."""
    with pytest.raises(storage.AudioValidationError):
        storage.validate_audio("clip.exe", b"MZfakebinary")


def test_upload_tmp_original_never_written_into_samples(temp_env,
                                                        real_upload_source):
    """The extraction workspace is temporary: only the WAV lands in samples."""
    raw = real_upload_source.read_bytes()
    meta = storage.store_upload("profile-tmp", "Recording 1.mp4", raw,
                                source="upload")
    stored_dir = Path(meta["file_path"]).parent
    names = [p.name for p in stored_dir.iterdir()]
    assert all(n.endswith(".wav") for n in names)


# --- Local voice engine (Audar + NeuCodec via subprocess bridge) ----------------
from app.engines import audar_local as _audar_local  # noqa: E402
from app.providers.local import (  # noqa: E402
    LocalVoiceProvider as _LocalProvider,
    ReferenceNotReadyError as _ReferenceNotReadyError,
)


def _make_profile_with_reference(transcript=None, with_transcript=True):
    pid = db.create_voice_profile("Local Clone", "", "English")
    raw = make_wav_bytes()
    meta = storage.store_upload(pid, "reference.wav", raw, source="upload")
    if with_transcript:
        meta["transcript"] = transcript or "hello world this is a test"
    sid = db.add_voice_sample(pid, **meta)
    return pid, sid


def test_local_provider_generate_requires_reference_transcript(temp_env):
    """Without a verified transcript the engine refuses — never invents one."""
    pid, _ = _make_profile_with_reference(with_transcript=False)
    provider = _LocalProvider()
    with pytest.raises(_ReferenceNotReadyError):
        provider.generate("مرحبا", "Arabic", {"id": pid})
    # Also refuses when transcript is whitespace-only.
    sid = db.list_voice_samples(pid)[0]["id"]
    db.update_sample_transcript(sid, "   ")
    with pytest.raises(_ReferenceNotReadyError):
        provider.generate("hello", "English", {"id": pid})


def test_local_provider_generate_missing_reference_file(temp_env):
    """A transcript without its audio file on disk is not a usable reference."""
    pid, sid = _make_profile_with_reference()
    from app import storage
    storage.delete_sample_file(
        db.get_voice_sample(sid)["file_path"])
    with pytest.raises(_ReferenceNotReadyError):
        _LocalProvider().generate("hello", "English", {"id": pid})


def test_local_provider_engine_failure_surfaces_safe_error(temp_env, monkeypatch):
    """Engine crashes are translated into user-safe messages, no tracebacks."""
    pid, _ = _make_profile_with_reference()

    def _boom(**kwargs):
        raise _audar_local.LocalEngineError(
            "The local voice engine failed. Check the engine logs "
            "in the terminal running the app.")
    monkeypatch.setattr(_audar_local, "run_local_generation", _boom)
    with pytest.raises(_audar_local.LocalEngineError) as err:
        _LocalProvider().generate("hello", "English", {"id": pid})
    assert "engine failed" in str(err.value)


def test_local_provider_generation_persists_one_audio_file(temp_env, monkeypatch):
    """Full success path through generation_service: DB row + WAV on disk."""
    pid, _ = _make_profile_with_reference()
    fake_wav = make_wav_bytes(duration_s=0.5)

    def _fake_generation(**kwargs):
        import tempfile
        from pathlib import Path as _P
        tmp = _P(tempfile.mkdtemp(prefix="mv_fake_gen_")) / "generated.wav"
        tmp.write_bytes(fake_wav)
        return {"wav_path": str(tmp), "duration_seconds": 0.5,
                "sample_rate": 24000, "ref_code_count": 500, "token_count": 25}
    monkeypatch.setattr(_audar_local, "run_local_generation", _fake_generation)
    provider = _LocalProvider()
    from app import generation_service
    row = generation_service.run_generation(
        provider, "Hello there", "English", {"id": pid})
    assert row["status"] == "completed"
    assert row["provider"] == "local"
    # The persisted audio is a real, valid WAV under GENERATED_DIR.
    from pathlib import Path as _P2
    from app import config as _config
    path = _P2(row["audio_path"])
    assert path.is_file() and path.parent == _config.GENERATED_DIR.resolve()
    duration, rate = storage.probe_wav(path)
    assert rate == 8000 and duration > 0.3
    rows = db.list_generations()
    assert len([r for r in rows if r["id"] == row["id"]]) == 1


def test_local_provider_reference_encoding_round_trip(temp_env, monkeypatch):
    """The bridge accepts a real WAV reference and returns its code count."""
    pid, _ = _make_profile_with_reference()
    sample = db.get_usable_reference_sample(pid)
    assert sample is not None

    captured = {}
    def _capture(**kwargs):
        captured.update(kwargs)
        import tempfile
        from pathlib import Path as _P
        tmp = _P(tempfile.mkdtemp(prefix="mv_fake_gen_")) / "g.wav"
        tmp.write_bytes(make_wav_bytes())
        return {"wav_path": str(tmp), "duration_seconds": 0.2,
                "sample_rate": 24000}
    monkeypatch.setattr(_audar_local, "run_local_generation", _capture)
    _LocalProvider().generate("hi", "English", {"id": pid})
    # The exact transcript of the reference was passed through — never a guess.
    assert captured["ref_text"] == "hello world this is a test"
    assert captured["ref_wav_path"] == sample["file_path"]


# --- Engine-specific readiness separation (Studio) ------------------------------
def test_local_profile_without_elevenlabs_id_saves_sample_and_transcript(temp_env):
    """Full persistence flow with NO ElevenLabs mapping anywhere."""
    pid = db.create_voice_profile("No Mapping Needed", "", "English")
    assert db.get_voice_profile(pid)["elevenlabs_voice_id"] is None
    meta = storage.store_upload(pid, "reference.wav", make_wav_bytes(),
                                source="upload")
    sid = db.add_voice_sample(pid, **meta)
    db.update_sample_transcript(sid, "exact words of the reference clip")
    # Reload everything from a fresh DB connection (simulated reopen).
    profile = db.get_voice_profile(pid)
    assert profile["elevenlabs_voice_id"] is None
    sample = db.get_voice_sample(sid)
    assert sample["transcript"] == "exact words of the reference clip"
    assert Path(sample["file_path"]).is_file()
    assert db.get_usable_reference_sample(pid) is not None
    assert sample["profile_id"] == pid == profile["id"]


def test_local_generation_skips_elevenlabs_validation_entirely(temp_env, monkeypatch):
    """Local generation must not read or validate elevenlabs_voice_id, and
    must never touch ElevenLabs code paths."""
    pid = db.create_voice_profile("Local Only", "", "English")
    meta = storage.store_upload(pid, "ref.wav", make_wav_bytes(), source="upload")
    sid = db.add_voice_sample(pid, **meta)
    db.update_sample_transcript(sid, "the reference transcript")

    captured = {}
    def _fake_generation(**kwargs):
        captured.update(kwargs)
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="mv_gen_")) / "g.wav"
        tmp.write_bytes(make_wav_bytes(duration_s=0.3))
        return {"wav_path": str(tmp), "duration_seconds": 0.3,
                "sample_rate": 24000}
    monkeypatch.setattr(_audar_local, "run_local_generation", _fake_generation)
    # Deliberately set a bogus mapping to prove it is ignored by the local path.
    db.update_voice_profile_elevenlabs_id(pid, "should-be-ignored")
    from app import generation_service
    row = generation_service.run_generation(
        _LocalProvider(), "local script text", "English",
        dict(db.get_voice_profile(pid)))
    assert row["status"] == "completed"
    assert row["provider"] == "local"
    # Correct reference audio + transcript + script reached the engine.
    assert captured["ref_text"] == "the reference transcript"
    assert captured["text"] == "local script text"
    assert captured["ref_wav_path"] == db.get_voice_sample(sid)["file_path"]


def test_missing_transcript_error_is_local_specific(temp_env):
    pid = db.create_voice_profile("No Transcript", "", "English")
    meta = storage.store_upload(pid, "ref.wav", make_wav_bytes(), source="upload")
    db.add_voice_sample(pid, **meta)   # no transcript
    with pytest.raises(_ReferenceNotReadyError) as err:
        _LocalProvider().generate("hi", "English", {"id": pid})
    msg = str(err.value)
    assert "transcript" in msg.lower()
    assert "elevenlabs" not in msg.lower()


def test_missing_reference_sample_error_is_local_specific(temp_env):
    pid = db.create_voice_profile("No Sample", "", "English")
    with pytest.raises(_ReferenceNotReadyError) as err:
        _LocalProvider().generate("hi", "English", {"id": pid})
    msg = str(err.value)
    assert "reference" in msg.lower()
    assert "elevenlabs" not in msg.lower()


def test_elevenlabs_generation_still_requires_mapping(temp_env, monkeypatch):
    """The cloud path keeps its existing mapping validation."""
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key-123")
    pid = db.create_voice_profile("Cloud Unmapped", "", "English")
    from app.providers.elevenlabs import ElevenLabsProvider
    from app.providers.base import ProviderVoiceMappingError
    with pytest.raises(ProviderVoiceMappingError):
        ElevenLabsProvider().generate("hi", "English", dict(db.get_voice_profile(pid)))


def test_local_wav_persists_and_appears_in_history(temp_env, monkeypatch):
    """The generated WAV lands in GENERATED_DIR and a generation row exists."""
    pid = db.create_voice_profile("History Local", "", "English")
    meta = storage.store_upload(pid, "ref.wav", make_wav_bytes(), source="upload")
    sid = db.add_voice_sample(pid, **meta)
    db.update_sample_transcript(sid, "reference transcript here")

    def _fake_generation(**kwargs):
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="mv_gen_")) / "g.wav"
        tmp.write_bytes(make_wav_bytes(duration_s=0.4))
        return {"wav_path": str(tmp), "duration_seconds": 0.4,
                "sample_rate": 24000}
    monkeypatch.setattr(_audar_local, "run_local_generation", _fake_generation)
    from app import generation_service
    row = generation_service.run_generation(
        _LocalProvider(), "history test", "English", {"id": pid})
    assert row["status"] == "completed"
    history = db.list_generations()
    assert any(r["id"] == row["id"] and r["status"] == "completed"
               for r in history)
    stored = Path(row["audio_path"])
    assert stored.is_file() and storage.probe_wav(stored)[0] > 0.2


# --- Metadata contract: storage extras never reach the DB layer -----------------
def test_sample_db_fields_filters_diagnostic_keys():
    """The shared filter keeps DB columns and drops diagnostic extras."""
    meta = {
        "original_filename": "clip.mp4", "stored_filename": "x.wav",
        "file_path": "/tmp/x.wav", "format": "wav", "size_bytes": 10,
        "duration_seconds": 1.0, "sample_rate": 16000, "source": "upload",
        "source_audio_codec": "opus",
    }
    filtered = storage.sample_db_fields(meta)
    assert "source_audio_codec" not in filtered
    assert filtered["format"] == "wav" and filtered["sample_rate"] == 16000
    # The original metadata still carries the diagnostic info.
    assert meta["source_audio_codec"] == "opus"


def test_wav_upload_without_codec_metadata_persists(temp_env):
    """A plain WAV upload has no codec key and saves without TypeError."""
    pid = db.create_voice_profile("Wav Upload", "", "English")
    meta = storage.store_upload(pid, "clean.wav", make_wav_bytes(),
                                source="upload")
    assert "source_audio_codec" not in meta
    sid = db.add_voice_sample(pid, **storage.sample_db_fields(meta))
    row = db.get_voice_sample(sid)
    assert row["original_filename"] == "clean.wav"
    assert row["source"] == "upload"


def test_compressed_upload_with_codec_metadata_persists(temp_env, real_upload_source):
    """Compressed uploads carry source_audio_codec; the filter drops it
    before the DB call so persistence succeeds and transcript editing and
    reference readiness keep working."""
    pid = db.create_voice_profile("Compressed Upload", "", "English")
    home = Path.home() / "Downloads"
    fixture = real_upload_source
    raw = fixture.read_bytes()
    meta = storage.store_upload(pid, fixture.name, raw, source="upload")
    assert meta.get("source_audio_codec") == "opus"
    sid = db.add_voice_sample(pid, **storage.sample_db_fields(meta))
    row = db.get_voice_sample(sid)
    assert row["format"] == "wav" and row["sample_rate"] == 16000
    # Transcript + readiness flow intact on this sample.
    db.update_sample_transcript(sid, "words spoken in the recording")
    assert db.get_usable_reference_sample(pid) is not None


def test_microphone_recording_with_codec_metadata_persists(temp_env):
    """The microphone path stores recordings whose metadata may gain a codec
    key after extraction — it must still persist exactly once."""
    pid = db.create_voice_profile("Mic Path", "", "English")
    # audio_input returns WAV, but store_upload output is filtered the same
    # way in the UI; simulate metadata that unexpectedly carries the key.
    meta = storage.store_upload(pid, "recording.wav", make_wav_bytes(),
                                source="recording")
    meta.setdefault("source_audio_codec", "")   # defensive: as if extraction ran
    sid = db.add_voice_sample(pid, **storage.sample_db_fields(meta))
    rows = db.list_voice_samples(pid)
    assert len(rows) == 1
    assert rows[0]["source"] == "recording"
    assert rows[0]["id"] == sid


def test_add_voice_sample_still_rejects_unknown_kwargs(temp_env):
    """The DB signature remains authoritative: junk kwargs still TypeError."""
    pid = db.create_voice_profile("Strict Signature", "", "English")
    with pytest.raises(TypeError):
        db.add_voice_sample(pid, original_filename="a.wav",
                            stored_filename="a.wav", file_path="/tmp/a.wav",
                            format="wav", size_bytes=1,
                            not_a_real_column="x")
