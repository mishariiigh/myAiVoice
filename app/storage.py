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
from . import media as media_mod

_MAX_MAGIC_BYTES = 4096

# Magic-byte signatures checked at the start of the file
_MPEG_SIGS = (b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2", b"\xff\xe3")

# Container formats accepted for upload but requiring FFmpeg extraction to
# become a voice-ready WAV. The extension never decides this alone: magic
# bytes are checked and the streams themselves are probed by app.media.
MEDIA_EXTENSIONS = {".mp4", ".m4a", ".aac", ".flac", ".ogg", ".opus",
                    ".webm", ".mov", ".mkv", ".wav", ".mp3"}


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
    if head[:4] == b"OggS":
        return "ogg"
    if head.startswith(b"fLaC"):
        return "flac"
    if head.startswith(b"\x1aE\xdf\xa3"):
        return "webm"   # Matroska/WebM (EBML)
    if head.startswith((b"\xff\xf1", b"\xff\xf9", b"\xff\xf8")):
        return "aac"    # ADTS AAC
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
            "File content does not look like a supported audio or video "
            "format (WAV, MP3, M4A/MP4, AAC, FLAC, OGG/Opus, WebM)."
        )

    # Cross-check extension against the real content type. The MP4/M4A
    # family shares one magic signature ('ftyp'), so .mp4 vs .m4a naming is
    # interchangeable at sniff level — both go through FFmpeg extraction.
    ext = Path(safe_name).suffix.lower()
    fmt_family = {"mp4": "m4a", "mov": "m4a", "opus": "ogg"}.get(
        ext.lstrip("."), ext.lstrip("."))
    if ext and fmt_family != fmt:
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


# Columns accepted by db.add_voice_sample(). Storage metadata may carry
# extra diagnostic keys (e.g. source_audio_codec from FFmpeg extraction);
# those are preserved in storage metadata but must never reach the DB layer,
# whose schema and signature are the single source of truth here.
_SAMPLE_DB_FIELDS = (
    "original_filename", "stored_filename", "file_path", "format",
    "size_bytes", "duration_seconds", "sample_rate", "source",
    "sample_kind", "transcript",
)


def sample_db_fields(meta: dict) -> dict:
    """Filter storage metadata down to the fields db.add_voice_sample accepts.

    Extra diagnostic keys (source_audio_codec, ...) are dropped from the
    returned dict but remain available in the original metadata dict for
    logging/UI. Unknown keys are ignored deliberately: the DB signature is
    authoritative, and passing anything else is a TypeError in waiting.
    """
    return {k: v for k, v in meta.items() if k in _SAMPLE_DB_FIELDS}


def store_upload(profile_id: str, filename: str, raw: bytes,
                 source: str = "upload") -> dict:
    """Validate and persist a recording for a profile.

    Returns metadata dict: keys stored_filename, file_path, format,
    size_bytes, duration_seconds, sample_rate, original_filename.
    """
    safe_name, fmt, duration, rate = validate_audio(filename, raw)

    ext = Path(safe_name).suffix.lower()
    extraction_meta = None
    stored_bytes = raw
    if fmt != "wav" or ext == ".mp4":
        # Container or compressed formats are decoded to a clean WAV
        # reference before storage. The raw upload is never modified:
        # it is written to a private temp file that FFmpeg only reads.
        if ext not in MEDIA_EXTENSIONS:
            raise AudioValidationError(
                f"Format '{fmt}' needs extraction but the extension "
                f"'{ext}' is not accepted."
            )
        import tempfile
        with tempfile.TemporaryDirectory(prefix="mv_upload_") as tmpdir:
            original = Path(tmpdir) / ("original" + ext)
            original.write_bytes(raw)
            extracted = Path(tmpdir) / "reference.wav"
            try:
                extraction_meta = media_mod.extract_reference_wav(
                    original, extracted)
            except media_mod.MediaExtractionError as exc:
                raise AudioValidationError(str(exc))
            stored_bytes = extracted.read_bytes()
            duration = extraction_meta.get("duration_seconds") or duration
            rate = extraction_meta.get("output_sample_rate") or rate
        safe_name = str(Path(safe_name).with_suffix(".wav"))
        fmt = "wav"

    target = _check_within_data(unique_target(config.SAMPLES_DIR, safe_name))
    target.write_bytes(stored_bytes)
    meta = {
        "original_filename": os.path.basename(filename),
        "stored_filename": target.name,
        "file_path": str(target),
        "format": fmt,
        "size_bytes": len(stored_bytes),
        "duration_seconds": duration,
        "sample_rate": rate,
        "source": source,
    }
    if extraction_meta:
        meta["source_audio_codec"] = extraction_meta.get("audio_codec") or ""
    return meta


def delete_sample_file(file_path: str) -> None:
    """Delete a stored sample after confirming it lives under DATA_DIR."""
    target = _check_within_data(Path(file_path))
    target.unlink(missing_ok=True)


# --- Generated audio -----------------------------------------------------------
#
# Generated speech is written to GENERATED_DIR, never to SAMPLES_DIR, so it
# stays logically separate from user-uploaded voice samples.

def validate_generated_audio(audio_bytes: bytes, declared_format: str) -> str:
    """Validate generated audio bytes before permanent storage.

    Checks:
    - Non-empty payload.
    - Size within the configured generated-audio limit.
    - Declared format matches the content sniffed from magic bytes.

    Returns the detected format string ('wav', 'mp3', or 'm4a').
    Raises AudioValidationError on any failure.
    """
    if not audio_bytes:
        raise AudioValidationError("Generated audio is empty.")
    max_bytes = config.MAX_GENERATED_MB * 1024 * 1024
    if len(audio_bytes) > max_bytes:
        raise AudioValidationError(
            f"Generated audio exceeds the {config.MAX_GENERATED_MB} MB limit "
            f"({len(audio_bytes)} bytes)."
        )
    detected = sniff_format(audio_bytes)
    if not detected:
        raise AudioValidationError(
            "Generated audio content does not look like WAV, MP3, or M4A audio."
        )
    if declared_format != detected:
        raise AudioValidationError(
            f"Declared format '{declared_format}' does not match detected "
            f"format '{detected}'."
        )
    return detected


def store_generated_audio(audio_bytes: bytes, fmt: str,
                          generation_id: str) -> str:
    """Validate and persist generated audio under GENERATED_DIR.

    The filename embeds the generation ID and is prefixed with a random
    digest, so collisions are cryptographically impossible and existing files
    are never overwritten.

    Returns the absolute path to the stored file.
    """
    detected = validate_generated_audio(audio_bytes, fmt)
    safe_name = f"gen_{generation_id}.{detected}"
    target = _check_within_data(unique_target(config.GENERATED_DIR, safe_name))
    target.write_bytes(audio_bytes)
    return str(target)


def delete_generated_file(file_path: str) -> None:
    """Delete a generated audio file after confirming it lives under
    GENERATED_DIR. Never deletes anything outside that directory."""
    resolved = Path(file_path).resolve()
    generated = config.GENERATED_DIR.resolve()
    if not str(resolved).startswith(str(generated)):
        raise AudioValidationError(
            "Refusing to delete a file outside the generated audio directory."
        )
    resolved.unlink(missing_ok=True)
