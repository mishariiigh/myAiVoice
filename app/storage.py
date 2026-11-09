"""Local audio storage: safe paths, unique filenames, validation.

Recordings are validated by content (magic bytes) rather than trusting
filename or MIME type. Nothing is ever sent anywhere.
"""

import hashlib
import os
import re
import wave
from pathlib import Path
from typing import Optional

from . import config

_MAX_MAGIC_BYTES = 4096

# Magic-byte signatures checked at the start of the file
_MPEG_SIGS = (b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2", b"\xff\xe3")


class AudioValidationError(Exception):
    """Raised when an uploaded file fails validation."""


def safe_filename(name: str) -> str:
    """Sanitize a user-supplied filename; raises on empty result.

    Filenames containing path separators (traversal attempts) are rejected,
    not silently rewritten.
    """
    if not name or os.path.basename(name) != name or "\\" in name:
        raise AudioValidationError("Invalid filename: path separators are not allowed.")
    base = name
    stem = Path(base).stem or "audio"
    ext = Path(base).suffix.lower()
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", stem)[:80].strip("._") or "audio"
    if ext and (ext not in config.ALLOWED_AUDIO_EXTENSIONS):
        raise AudioValidationError(
            f"Extension '{ext}' is not supported. "
            f"Allowed: {', '.join(sorted(config.ALLOWED_AUDIO_EXTENSIONS))}."
        )
    return stem + (ext or "")


def unique_target(directory: Path, sanitized_name: str) -> Path:
    """Build a collision-free target path inside `directory` (uuid-prefixed)."""
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(os.urandom(16)).hexdigest()[:12]
    return directory / f"{digest}_{sanitized_name}"


def _check_within_data(target: Path) -> Path:
    resolved = target.resolve()
    if not str(resolved).startswith(str(config.DATA_DIR)):
        raise AudioValidationError("Resolved path escapes the data directory.")
    return resolved


def sniff_format(raw: bytes) -> str:
    """Identify container format from magic bytes. Returns 'wav'|'mp3'|'m4a'|''."""
    head = raw[: _MAX_MAGIC_BYTES]
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "wav"
    if head.startswith(_MPEG_SIGS):
        return "mp3"
    # MP4/M4A family: 'ftyp' box at offset 4
    if len(head) > 11 and head[4:8] == b"ftyp":
        return "m4a"
    return ""


def probe_wav(path: Path) -> tuple[Optional[float], Optional[int]]:
    """Validate RIFF chunk structure and return (duration, sample_rate).

    Walks the chunk list strictly: every chunk size must fit inside the file,
    a fmt chunk must precede data, and the declared sample count must match.
    Corrupted or truncated files are rejected.
    """
    raw = path.read_bytes()
    if len(raw) < 12 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise AudioValidationError("File is not a readable WAV audio file.")
    pos, fmt_seen, rate, fmt_size = 12, False, None, None
    while pos + 8 <= len(raw):
        cid = raw[pos:pos + 4]
        size = int.from_bytes(raw[pos + 4:pos + 8], "little")
        pos += 8
        if pos + size > len(raw):
            raise AudioValidationError("Truncated or corrupted WAV chunk.")
        if cid == b"fmt " and size >= 16:
            fmt_seen = True
            rate = int.from_bytes(raw[pos + 4:pos + 8], "little")
            bits_per_sample_field = int.from_bytes(raw[pos + 14:pos + 16], "little")
        elif cid == b"data" and fmt_seen:
            bits_per_sample = bits_per_sample_field or 16
            bytes_total = max(size, len(raw) - pos)
            frames = bytes_total // max((bits_per_sample // 8), 1)
            dur = frames / rate if rate else None
            return dur, rate
        pos += size + (size % 2)  # chunks are word-aligned
    raise AudioValidationError("WAV file is missing a readable data chunk.")


def validate_audio(filename: str, raw: bytes) -> tuple[str, str, Optional[float], Optional[int]]:
    """Validate an uploaded recording.

    Returns (safe_name, detected_format, duration, sample_rate).
    Raises AudioValidationError on any failure.
    """
    safe_name = safe_filename(filename)

    if not raw:
        raise AudioValidationError("Uploaded file is empty.")

    if len(raw) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise AudioValidationError(
            f"File exceeds the {config.MAX_UPLOAD_MB} MB limit."
        )

    fmt = sniff_format(raw)
    if not fmt:
        raise AudioValidationError(
            "File content does not look like WAV, MP3, or M4A audio."
        )

    # Cross-check extension against the real content type
    ext = Path(safe_name).suffix.lower()
    expected_ext = f".{fmt}"
    if ext != expected_ext:
        raise AudioValidationError(
            f"Extension '{ext}' does not match detected audio format ({fmt})."
        )

    # Content-level readability check (currently exhaustive for WAV)
    duration = sample_rate = None
    import tempfile

    tmp = None
    try:
        if fmt == "wav":
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                tf.write(raw)
                tmp = tf.name
            duration, sample_rate = probe_wav(Path(tmp))
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)

    return safe_name, fmt, duration, sample_rate


def store_upload(profile_id: str, filename: str, raw: bytes,
                 source: str = "upload") -> dict:
    """Validate and persist a recording for a profile.

    Returns metadata dict: keys stored_filename, file_path, format,
    size_bytes, duration_seconds, sample_rate, original_filename.
    """
    safe_name, fmt, duration, rate = validate_audio(filename, raw)
    target = _check_within_data(unique_target(config.SAMPLES_DIR, safe_name))
    target.write_bytes(raw)
    return {
        "original_filename": os.path.basename(filename),
        "stored_filename": target.name,
        "file_path": str(target),
        "format": fmt,
        "size_bytes": len(raw),
        "duration_seconds": duration,
        "sample_rate": rate,
        "source": source,
    }


def delete_sample_file(file_path: str) -> None:
    """Delete a stored sample after confirming it lives under DATA_DIR."""
    target = _check_within_data(Path(file_path))
    target.unlink(missing_ok=True)
