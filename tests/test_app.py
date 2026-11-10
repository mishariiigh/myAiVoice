"""Pytest coverage for Mishari Voice Studio Stage 1."""

import sqlite3
import stat
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
    monkeypatch.setattr(config, "ELEVENLABS_API_KEY", "test-key-123")
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
