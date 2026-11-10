"""SQLite persistence layer for Mishari Voice Studio.

Initialization is idempotent: existing databases are never dropped or
recreated. All data-modifying helpers use transactions.
"""

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from . import config

SCHEMA_VERSION = 2

# Schema version 1 — the baseline tables that existed before any migration.
# Every statement is CREATE TABLE/INDEX IF NOT EXISTS so re-running it on an
# already-migrated database is a no-op that preserves all existing data.
_V1_SCHEMA = """
CREATE TABLE IF NOT EXISTS voice_profiles (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    preferred_language TEXT NOT NULL DEFAULT 'English'
        CHECK (preferred_language IN ('English', 'Arabic')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS voice_samples (
    id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL REFERENCES voice_profiles(id) ON DELETE CASCADE,
    original_filename TEXT NOT NULL,
    stored_filename TEXT NOT NULL UNIQUE,
    file_path TEXT NOT NULL UNIQUE,
    format TEXT NOT NULL,
    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
    duration_seconds REAL,
    sample_rate INTEGER,
    source TEXT NOT NULL DEFAULT 'upload' CHECK (source IN ('upload', 'recording')),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scripts (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    language TEXT NOT NULL CHECK (language IN ('English', 'Arabic')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS generations (
    id TEXT PRIMARY KEY,
    script_text TEXT NOT NULL,
    language TEXT NOT NULL CHECK (language IN ('English', 'Arabic')),
    provider TEXT NOT NULL,
    provider_model TEXT,
    voice_profile_id TEXT REFERENCES voice_profiles(id) ON DELETE SET NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    error_message TEXT,
    audio_file_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audio_files (
    id TEXT PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    format TEXT NOT NULL,
    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
    duration_seconds REAL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_samples_profile ON voice_samples(profile_id);
CREATE INDEX IF NOT EXISTS idx_generations_created ON generations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_generations_profile ON generations(voice_profile_id);
CREATE INDEX IF NOT EXISTS idx_scripts_created ON scripts(created_at DESC);
"""

# Schema version 2 additions — applied as an idempotent migration on top of v1.
# Every statement is safe to run against an already-migrated database.
_V2_MIGRATIONS = [
    # Distinguish voice-identity samples from expressive reference clips.
    # Existing rows default to 'identity' so nothing is lost or reclassified
    # silently; only NEW samples must opt into being expression references.
    "ALTER TABLE voice_samples ADD COLUMN sample_kind TEXT NOT NULL DEFAULT 'identity' "
    "CHECK (sample_kind IN ('identity', 'expression'))",

    # Expressive reference library. audio_file_id is deliberately NULLABLE:
    # a reference without a usable audio clip must not be selectable for
    # generation, but it may still exist as a metadata placeholder (e.g. a
    # recording that failed validation). ON DELETE SET NULL is only valid on
    # a nullable column — NOT NULL + SET NULL would be a contradiction.
    """
    CREATE TABLE IF NOT EXISTS expression_references (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT NOT NULL DEFAULT '',
        emotion TEXT NOT NULL,
        language TEXT NOT NULL CHECK (language IN ('English', 'Arabic')),
        audio_file_id TEXT REFERENCES audio_files(id) ON DELETE SET NULL,
        notes TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_expr_ref_emotion ON expression_references(emotion)",
    "CREATE INDEX IF NOT EXISTS idx_expr_ref_lang ON expression_references(language)",
    "CREATE INDEX IF NOT EXISTS idx_expr_ref_created ON expression_references(created_at DESC)",

    # Record which emotion / expressive reference was used for each generation.
    # provider_model already exists from v1 — it is intentionally NOT re-added.
    "ALTER TABLE generations ADD COLUMN emotion TEXT",
    "ALTER TABLE generations ADD COLUMN expression_reference_id TEXT "
    "REFERENCES expression_references(id) ON DELETE SET NULL",
    "CREATE INDEX IF NOT EXISTS idx_generations_emotion ON generations(emotion)",
    "CREATE INDEX IF NOT EXISTS idx_generations_expr_ref ON generations(expression_reference_id)",

    # Persist the applied schema version so future migrations can be gated.
    """
    CREATE TABLE IF NOT EXISTS schema_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
]


# Sentinel default for "field not provided" in update_* helpers. A bare None
# would mean "set the column to NULL", so the absence of a change needs its own
# distinct marker (see update_expression_reference audio_file_id handling).
_UNSET = object()


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _table_columns(conn: sqlite3.Connection, table: str) -> set:
    """Return the set of column names for `table`, or an empty set if it does
    not exist. Used to detect which migrations have already been applied."""
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    except sqlite3.OperationalError:
        return set()
    return {r["name"] for r in rows}


def _get_schema_version(conn: sqlite3.Connection) -> int:
    """Return the recorded schema version, or 0 when unknown/unmigrated.

    A missing `schema_meta` table — the ONLY condition the SELECT raises
    "no such table: schema_meta" for — is treated as "unmigrated". Any
    other OperationalError (database locked, disk full, ...) is unrelated
    and is deliberately allowed to propagate rather than being silently
    treated as "version 0".
    """
    try:
        row = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()
    except sqlite3.OperationalError as exc:
        if "no such table: schema_meta" in str(exc):
            return 0
        raise
    if row is None:
        return 0
    try:
        return int(row["value"])
    except (TypeError, ValueError):
        return 0


def _add_column_sql(statement: str) -> Optional[tuple[str, str]]:
    """Return (table, column) if `statement` is an `ALTER TABLE ... ADD COLUMN`,
    else None. Used to skip re-adding a column that already exists."""
    tokens = statement.split()
    if len(tokens) >= 5 and tokens[0].upper() == "ALTER" \
            and tokens[1].upper() == "TABLE" \
            and tokens[2] not in ("IF",) \
            and tokens[3].upper() == "ADD" and tokens[4].upper() == "COLUMN":
        return tokens[2], tokens[5]
    return None


def _set_schema_version(conn: sqlite3.Connection, version: int) -> None:
    conn.execute(
        "INSERT INTO schema_meta (key, value, updated_at) VALUES (?, ?, ?)"
        " ON CONFLICT(key) DO UPDATE SET value = excluded.value,"
        " updated_at = excluded.updated_at",
        ("schema_version", str(version), _utcnow()),
    )


def init_db(db_path: Optional[Path] = None) -> None:
    """Create tables/indexes if missing and apply pending migrations.

    Idempotent and safe to call repeatedly. Existing data is never dropped or
    recreated.

    Transaction behavior:
    - The v1 baseline runs via `executescript()` (which manages its own
      transaction and commits on its own) — no `executescript()` is ever used
      inside the v2 migration transaction.
    - The v2 migration (and the schema-version recording that gates it) run
      as explicit, individually-executed statements inside a
      `BEGIN IMMEDIATE` transaction: it commits only after every statement,
      including `_set_schema_version`, has succeeded, and rolls back
      completely if any statement fails, so the schema version can never
      advance past a partially-applied migration.
    - `ALTER TABLE ... ADD COLUMN` statements are guarded by a column
      existence check so a database that was somehow left partially migrated
      (columns added but the version row missing) can never be re-ALTERed
      into a "duplicate column" failure.
    """
    with _connect(db_path) as conn:
        # 1. Baseline schema (v1). CREATE TABLE IF NOT EXISTS is a no-op when
        #    the tables already exist, so this preserves all existing data.
        conn.executescript(_V1_SCHEMA)

        # 2. Apply v2 migrations only if the database has not recorded it yet,
        #    as one explicit, all-or-nothing transaction.
        current = _get_schema_version(conn)
        if current < SCHEMA_VERSION:
            if conn.in_transaction:
                # Defensive: executescript() commits on its own, but never
                # BEGIN with one transaction already open.
                conn.commit()
            conn.execute("BEGIN IMMEDIATE")
            try:
                for statement in _V2_MIGRATIONS:
                    add_column = _add_column_sql(statement)
                    if add_column and add_column[1] in _table_columns(conn, add_column[0]):
                        continue  # already added — do not re-ALTER
                    conn.execute(statement)
                _set_schema_version(conn, SCHEMA_VERSION)
            except Exception:
                conn.rollback()
                raise
            conn.commit()


@contextmanager
def get_conn(db_path: Optional[Path] = None):
    conn = _connect(db_path)
    try:
        with conn:  # transaction context: commits or rolls back as a unit
            yield conn
    finally:
        conn.close()


def new_id() -> str:
    return uuid.uuid4().hex


# --- Voice profiles -----------------------------------------------------------

def create_voice_profile(name: str, description: str = "",
                         preferred_language: str = "English") -> str:
    pid = new_id()
    now = _utcnow()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO voice_profiles (id, name, description, preferred_language,"
            " created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (pid, name.strip(), description.strip(), preferred_language, now, now),
        )
    return pid


def update_voice_profile(profile_id: str, name: Optional[str] = None,
                         description: Optional[str] = None,
                         preferred_language: Optional[str] = None) -> None:
    fields, params = [], []
    if name is not None:
        fields.append("name = ?"); params.append(name.strip())
    if description is not None:
        fields.append("description = ?"); params.append(description.strip())
    if preferred_language is not None:
        fields.append("preferred_language = ?"); params.append(preferred_language)
    if not fields:
        return
    fields.append("updated_at = ?"); params.append(_utcnow())
    params.append(profile_id)
    with get_conn() as conn:
        conn.execute(f"UPDATE voice_profiles SET {', '.join(fields)} WHERE id = ?", params)


def delete_voice_profile(profile_id: str) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM voice_profiles WHERE id = ?", (profile_id,))


def get_voice_profile(profile_id: str) -> Optional[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM voice_profiles WHERE id = ?", (profile_id,)
        ).fetchone()


def list_voice_profiles() -> list:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM voice_profiles ORDER BY created_at DESC"
        ).fetchall()


# --- Voice samples -------------------------------------------------------------

def add_voice_sample(profile_id: str, original_filename: str, stored_filename: str,
                      file_path: str, format: str, size_bytes: int,
                      duration_seconds: Optional[float] = None,
                      sample_rate: Optional[int] = None,
                      source: str = "upload",
                      sample_kind: str = "identity") -> str:
    sid = new_id()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO voice_samples (id, profile_id, original_filename,"
            " stored_filename, file_path, format, size_bytes, duration_seconds,"
            " sample_rate, source, sample_kind, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (sid, profile_id, original_filename, stored_filename, file_path, format,
             size_bytes, duration_seconds, sample_rate, source, sample_kind,
             _utcnow()),
        )
    return sid


def get_voice_sample(sample_id: str) -> Optional[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM voice_samples WHERE id = ?", (sample_id,)
        ).fetchone()


def list_voice_samples(profile_id: str) -> list:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM voice_samples WHERE profile_id = ? ORDER BY created_at DESC",
            (profile_id,),
        ).fetchall()


def delete_voice_sample(sample_id: str) -> None:
    """Remove metadata and the file on disk (inside a transaction)."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT file_path FROM voice_samples WHERE id = ?", (sample_id,)
        ).fetchone()
        if row:
            conn.execute("DELETE FROM voice_samples WHERE id = ?", (sample_id,))
            Path(row["file_path"]).unlink(missing_ok=True)


# --- Generations ----------------------------------------------------------------

def add_generation(script_text: str, language: str, provider: str,
                    voice_profile_id: Optional[str] = None,
                    provider_model: Optional[str] = None,
                    status: str = "pending",
                    error_message: Optional[str] = None,
                    emotion: Optional[str] = None,
                    expression_reference_id: Optional[str] = None) -> str:
    gid = new_id()
    now = _utcnow()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO generations (id, script_text, language, provider,"
            " provider_model, voice_profile_id, status, error_message,"
            " emotion, expression_reference_id, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (gid, script_text, language, provider, provider_model,
             voice_profile_id, status, error_message, emotion,
             expression_reference_id, now, now),
        )
    return gid


def update_generation(generation_id: str, status: Optional[str] = None,
                      error_message: Optional[str] = None,
                      audio_file_id: Optional[str] = None) -> None:
    fields, params = [], []
    if status is not None:
        fields.append("status = ?"); params.append(status)
    if error_message is not None:
        fields.append("error_message = ?"); params.append(error_message)
    if audio_file_id is not None:
        fields.append("audio_file_id = ?"); params.append(audio_file_id)
    if not fields:
        return
    fields.append("updated_at = ?"); params.append(_utcnow())
    params.append(generation_id)
    with get_conn() as conn:
        conn.execute(f"UPDATE generations SET {', '.join(fields)} WHERE id = ?", params)


def list_generations() -> list:
    with get_conn() as conn:
        return conn.execute(
            "SELECT g.*, p.name AS voice_profile_name, a.path AS audio_path,"
            " r.name AS expression_reference_name"
            " FROM generations g"
            " LEFT JOIN voice_profiles p ON p.id = g.voice_profile_id"
            " LEFT JOIN audio_files a ON a.id = g.audio_file_id"
            " LEFT JOIN expression_references r ON r.id = g.expression_reference_id"
            " ORDER BY g.created_at DESC"
        ).fetchall()


# --- Audio files -----------------------------------------------------------------

def add_audio_file(path: str, fmt: str, size_bytes: int,
                   duration_seconds: Optional[float] = None) -> str:
    aid = new_id()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO audio_files (id, path, format, size_bytes, duration_seconds,"
            " created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (aid, path, fmt, size_bytes, duration_seconds, _utcnow()),
        )
    return aid


def list_audio_files() -> list:
    with get_conn() as conn:
        return conn.execute("SELECT * FROM audio_files ORDER BY created_at DESC").fetchall()


# --- Settings (key/value) ----------------------------------------------------------

def get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    with get_conn() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default


def set_setting(key: str, value: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value,"
            " updated_at = excluded.updated_at",
            (key, value, _utcnow()),
        )


def list_settings() -> list:
    with get_conn() as conn:
        return conn.execute("SELECT * FROM settings ORDER BY key").fetchall()


# --- Scripts -------------------------------------------------------------------------

def save_script(title: str, content: str, language: str) -> str:
    sid = new_id()
    now = _utcnow()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO scripts (id, title, content, language, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (sid, title, content, language, now, now),
        )
    return sid


def list_scripts() -> list:
    with get_conn() as conn:
        return conn.execute("SELECT * FROM scripts ORDER BY created_at DESC").fetchall()


# --- Expression references --------------------------------------------------------
#
# Expressive reference clips are logically separate from voice-identity samples:
# they describe HOW the generated speech should be delivered (emotion, pacing,
# tone) rather than WHO is speaking. They reuse the audio_files table for the
# actual recording so the same validated storage path is used for both kinds
# of audio.

def create_expression_reference(name: str, emotion: str, language: str,
                                description: str = "",
                                audio_file_id: Optional[str] = None,
                                notes: str = "") -> str:
    """Insert a new expressive reference.

    `audio_file_id` is optional — a reference may be created as a metadata
    placeholder (e.g. while a recording is being validated) and linked to an
    audio file later. Callers must check `audio_file_id` before selecting the
    reference for generation: a reference without a usable audio clip is not a
    valid choice.
    """
    rid = new_id()
    now = _utcnow()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO expression_references"
            " (id, name, description, emotion, language, audio_file_id, notes,"
            " created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (rid, name.strip(), description.strip(), emotion, language,
             audio_file_id, notes.strip(), now, now),
        )
    return rid


def get_expression_reference(reference_id: str) -> Optional[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM expression_references WHERE id = ?", (reference_id,)
        ).fetchone()


def list_expression_references(emotion: Optional[str] = None,
                                language: Optional[str] = None,
                                usable_only: bool = True) -> list:
    """List expressive references, optionally filtered by emotion/language.

    When `usable_only` is True (the default), only references that have a
    valid audio_file_id are returned — a reference without a usable recording
    must not appear as a selectable generation option.
    """
    where, params = [], []
    if emotion is not None:
        where.append("emotion = ?"); params.append(emotion)
    if language is not None:
        where.append("language = ?"); params.append(language)
    if usable_only:
        where.append("audio_file_id IS NOT NULL")
    sql = "SELECT * FROM expression_references"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY emotion, created_at DESC"
    with get_conn() as conn:
        return conn.execute(sql, params).fetchall()


def update_expression_reference(reference_id: str,
                                name: Optional[str] = None,
                                description: Optional[str] = None,
                                emotion: Optional[str] = None,
                                language: Optional[str] = None,
                                audio_file_id=_UNSET,
                                notes: Optional[str] = None) -> None:
    """Update mutable fields on an expressive reference.

    `audio_file_id` is tri-state because callers need three distinct actions:
    - omit the argument (default)  -> no change.
    - pass an audio file id (str)  -> assign a new reference.
    - pass None explicitly         -> clear the link (e.g. the referenced
      audio was deleted or its recording failed validation).
    """
    fields, params = [], []
    if name is not None:
        fields.append("name = ?"); params.append(name.strip())
    if description is not None:
        fields.append("description = ?"); params.append(description.strip())
    if emotion is not None:
        fields.append("emotion = ?"); params.append(emotion)
    if language is not None:
        fields.append("language = ?"); params.append(language)
    if audio_file_id is not _UNSET:
        # None clears the link; a str assigns a new reference.
        fields.append("audio_file_id = ?"); params.append(audio_file_id)
    if notes is not None:
        fields.append("notes = ?"); params.append(notes.strip())
    if not fields:
        return
    fields.append("updated_at = ?"); params.append(_utcnow())
    params.append(reference_id)
    with get_conn() as conn:
        conn.execute(
            f"UPDATE expression_references SET {', '.join(fields)}"
            " WHERE id = ?", params,
        )


def delete_expression_reference(reference_id: str) -> None:
    """Delete an expressive reference, and its audio file ONLY if nothing else
    still uses it.

    Safety model:
    - Other expression references and generations are checked inside the
      database transaction; if any of them references the audio file, the
      audio row and the physical file are preserved (referential integrity).
    - The database transaction is committed BEFORE the physical file is
      deleted, so a database error can never leave an orphaned file, and a
      filesystem failure can never leave a dangling foreign key.
    """
    with get_conn() as conn:
        row = conn.execute(
            "SELECT audio_file_id FROM expression_references WHERE id = ?",
            (reference_id,),
        ).fetchone()
        if row is None:
            return
        audio_id = row["audio_file_id"]
        delete_audio = bool(
            audio_id
            and not conn.execute(
                "SELECT 1 FROM expression_references"
                " WHERE audio_file_id = ? AND id != ? LIMIT 1",
                (audio_id, reference_id),
            ).fetchone()
            and not conn.execute(
                "SELECT 1 FROM generations WHERE audio_file_id = ? LIMIT 1",
                (audio_id,),
            ).fetchone()
        )
        conn.execute("DELETE FROM expression_references WHERE id = ?", (reference_id,))
        audio_path = None
        if delete_audio:
            audio_row = conn.execute(
                "SELECT path FROM audio_files WHERE id = ?", (audio_id,)
            ).fetchone()
            if audio_row:
                conn.execute("DELETE FROM audio_files WHERE id = ?", (audio_id,))
                audio_path = audio_row["path"]
    # Transaction committed — the reference row is gone. Only now remove the
    # physical file, and only if the transaction said it was safe.
    if audio_path:
        Path(audio_path).unlink(missing_ok=True)
