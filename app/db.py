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

SCHEMA_VERSION = 1

_SCHEMA = """
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


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    """Create tables/indexes if missing. Never erases existing data."""
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA)


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
                     source: str = "upload") -> str:
    sid = new_id()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO voice_samples (id, profile_id, original_filename,"
            " stored_filename, file_path, format, size_bytes, duration_seconds,"
            " sample_rate, source, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (sid, profile_id, original_filename, stored_filename, file_path, format,
             size_bytes, duration_seconds, sample_rate, source, _utcnow()),
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
                   error_message: Optional[str] = None) -> str:
    gid = new_id()
    now = _utcnow()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO generations (id, script_text, language, provider,"
            " provider_model, voice_profile_id, status, error_message,"
            " created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (gid, script_text, language, provider, provider_model,
             voice_profile_id, status, error_message, now, now),
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
            "SELECT g.*, p.name AS voice_profile_name, a.path AS audio_path"
            " FROM generations g"
            " LEFT JOIN voice_profiles p ON p.id = g.voice_profile_id"
            " LEFT JOIN audio_files a ON a.id = g.audio_file_id"
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
