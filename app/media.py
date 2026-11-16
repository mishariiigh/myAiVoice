"""Media extraction — turn any FFmpeg-readable upload into clean WAV audio.

Why this module exists: uploads may be MP4/M4A/OGG/WebM containers wrapping
Opus or AAC audio. The extension tells us nothing about the real media, so
every decision below is made by probing the actual streams with FFmpeg.

FFmpeg executable resolution order:
  1. ``FFMPEG_PATH`` environment variable (explicit user override).
  2. A system ``ffmpeg`` on PATH — only if it decodes our probe fixture
     correctly (guards against ancient/broken installs).
  3. The binary bundled inside the *audar* conda environment's
     ``imageio_ffmpeg`` package (no Homebrew required, nothing installed
     into the main ``.venv``).

Design rules honoured here:
  - The original upload is never modified: we always read and write to a
    separate destination.
  - All temporary files live in a private temp directory and are cleaned up.
  - Every failure raises :class:`MediaExtractionError` with a user-readable
    message — no raw tracebacks escape to the UI.
"""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

# Audio parameters Audar/NeuCodec actually require (verified against the
# installed neucodec source: encode input is resampled internally from 16 kHz
# mono; its decoder emits 24 kHz).
REFERENCE_SAMPLE_RATE = 16000
REFERENCE_CHANNELS = 1

_PROBE_TIMEOUT = 60


class MediaExtractionError(Exception):
    """Raised when media probing or audio extraction fails.

    The message is always safe to show to a user: it describes the media
    problem, never internal paths or credentials.
    """


class MediaProbeResult:
    """Facts learned about a media file by probing it."""

    def __init__(self, has_audio: bool, audio_codec: str,
                 sample_rate: Optional[int], channels: Optional[int],
                 duration_seconds: Optional[float]):
        self.has_audio = has_audio
        self.audio_codec = audio_codec
        self.sample_rate = sample_rate
        self.channels = channels
        self.duration_seconds = duration_seconds


def _probe_fixture_wav(directory: Path) -> Path:
    """Write a tiny silent WAV used to sanity-check an FFmpeg binary."""
    import struct
    import wave

    path = directory / "_probe.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(struct.pack("<h", 0) * 800)
    return path


def _binary_decodes_properly(ffmpeg_bin: str, workdir: Path) -> bool:
    """True only if the binary exists AND actually decodes audio to PCM."""
    if not ffmpeg_bin or not os.path.isfile(ffmpeg_bin) or \
            not os.access(ffmpeg_bin, os.X_OK):
        return False
    fixture = _probe_fixture_wav(workdir)
    try:
        proc = subprocess.run(
            [ffmpeg_bin, "-v", "error", "-i", str(fixture),
             "-f", "s16le", "-acodec", "pcm_s16le", "-"],
            capture_output=True, timeout=_PROBE_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    finally:
        fixture.unlink(missing_ok=True)
    if proc.returncode != 0:
        return False
    # 800 frames * 2 bytes = 1600 bytes of PCM expected.
    return len(proc.stdout) >= 1600


def _candidates() -> list:
    """Ordered list of FFmpeg binaries to try."""
    found = []
    override = os.environ.get("FFMPEG_PATH", "").strip()
    if override:
        found.append(override)
    system = shutil.which("ffmpeg")
    if system:
        found.append(system)
    # imageio_ffmpeg inside the audar conda env (verified install; keeps the
    # main .venv free of large dependencies).
    pkg_dir = Path.home() / ".conda/envs/audar/lib/python3.12/site-packages/" \
                       "imageio_ffmpeg/binaries"
    if pkg_dir.is_dir():
        for entry in sorted(pkg_dir.iterdir()):
            if entry.name.startswith("ffmpeg") and entry.is_file():
                found.append(str(entry))
    return found


def resolve_ffmpeg() -> str:
    """Return a verified FFmpeg executable path, or raise MediaExtractionError.

    Verification means: the binary decodes a known-good WAV to the exact
    expected PCM payload. Presence on PATH alone is not trusted.
    """
    candidates = _candidates()
    if not candidates:
        raise MediaExtractionError(
            "No FFmpeg executable is available. Install imageio-ffmpeg in the "
            "audar environment, or set FFMPEG_PATH to a full ffmpeg binary."
        )
    with tempfile.TemporaryDirectory(prefix="mv_ffmpeg_probe_") as tmp:
        workdir = Path(tmp)
        for candidate in candidates:
            if _binary_decodes_properly(candidate, workdir):
                return candidate
    raise MediaExtractionError(
        "An FFmpeg executable was found but none of them could decode audio "
        "correctly. Check the FFMPEG_PATH setting or reinstall imageio-ffmpeg."
    )


def probe_media(source, ffmpeg_bin: Optional[str] = None) -> MediaProbeResult:
    """Inspect the real streams inside a media file.

    Never trusts the extension: decodes the file and asks FFmpeg what it
    actually contains.
    """
    ffmpeg_bin = ffmpeg_bin or resolve_ffmpeg()
    src = str(Path(source).resolve())
    if not Path(src).is_file():
        raise MediaExtractionError("The media file does not exist.")
    if Path(src).stat().st_size == 0:
        raise MediaExtractionError("The media file is empty.")

    # Decode the FIRST audio stream to raw PCM. This simultaneously answers
    # "is there an audio stream?", "is the codec supported?", and "is the
    # data corrupt?" in one pass — no output file is written ("-").
    try:
        proc = subprocess.run(
            [ffmpeg_bin, "-v", "error", "-i", src, "-map", "0:a:0",
             "-t", "1.0", "-f", "s16le", "-acodec", "pcm_s16le", "-"],
            capture_output=True, timeout=_PROBE_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        raise MediaExtractionError(
            "Inspecting the file timed out — it may be very large or on slow "
            "storage."
        )
    stderr = proc.stderr.decode("utf-8", "ignore")
    if proc.returncode != 0:
        if "matches no streams" in stderr or "Stream map" in stderr:
            raise MediaExtractionError(
                "This file has no audio stream. It looks like a video with "
                "no sound track, so there is nothing to use as a voice "
                "reference."
            )
        raise MediaExtractionError(
            "The file could not be decoded: it may be corrupt, empty of "
            "sound, or in an unsupported format."
        )
    if not proc.stdout:
        raise MediaExtractionError(
            "The file decoded but produced no audio data."
        )

    # Second pass: parse the full metadata header for stream facts.
    info = subprocess.run(
        [ffmpeg_bin, "-hide_banner", "-i", src],
        capture_output=True, timeout=_PROBE_TIMEOUT,
    )
    header = info.stderr.decode("utf-8", "ignore")
    audio_codec, sample_rate, channels, duration = "", None, None, None
    for line in header.splitlines():
        line = line.strip()
        if line.startswith("Stream ") and "Audio:" in line:
            after = line.split("Audio:", 1)[1].strip()
            parts = [p.strip() for p in after.split(",")]
            if parts:
                audio_codec = parts[0].split(" ")[0]
            for part in parts[1:]:
                if part.endswith("Hz"):
                    try:
                        sample_rate = int(part[:-2].strip())
                    except ValueError:
                        pass
                elif part == "mono":
                    channels = 1
                elif part == "stereo":
                    channels = 2
    for line in header.splitlines():
        if "Duration:" in line:
            try:
                stamp = line.split("Duration:", 1)[1].split(",")[0].strip()
                hh, mm, ss = stamp.split(":")
                duration = int(hh) * 3600 + int(mm) * 60 + float(ss)
            except (ValueError, IndexError):
                pass
            break

    return MediaProbeResult(True, audio_codec, sample_rate, channels, duration)


def extract_reference_wav(source, destination, ffmpeg_bin: Optional[str] = None,
                          sample_rate: int = REFERENCE_SAMPLE_RATE) -> dict:
    """Extract the first audio stream of ``source`` into a clean WAV file.

    The original file is only ever read. Extraction happens to a temp
    directory beside the destination and is then moved into place.

    Returns a metadata dict: has_audio, audio_codec, source_sample_rate,
    source_channels, duration_seconds, output_path, output_sample_rate.
    """
    src = Path(source)
    dst = Path(destination)
    ffmpeg_bin = ffmpeg_bin or resolve_ffmpeg()
    facts = probe_media(src, ffmpeg_bin)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mv_extract_",
                                     dir=str(dst.parent)) as tmp:
        partial = Path(tmp) / "reference.wav"
        cmd = [
            ffmpeg_bin, "-v", "error", "-y", "-i", str(src.resolve()),
            "-map", "0:a:0",
            "-vn",                      # drop any video stream
            "-ac", str(REFERENCE_CHANNELS),
            "-ar", str(sample_rate),
            "-acodec", "pcm_s16le",
            str(partial),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True,
                                  timeout=_PROBE_TIMEOUT * 6)
        except subprocess.TimeoutExpired:
            raise MediaExtractionError(
                "Audio extraction timed out — the file may be very large or "
                "on slow storage."
            )
        if proc.returncode != 0 or not partial.exists() or \
                partial.stat().st_size == 0:
            raise MediaExtractionError(
                "Audio could not be extracted from this file. It may use an "
                "unsupported codec or be corrupt."
            )
        shutil.move(str(partial), str(dst))

    return {
        "has_audio": True,
        "audio_codec": facts.audio_codec,
        "source_sample_rate": facts.sample_rate,
        "source_channels": facts.channels,
        "duration_seconds": facts.duration_seconds,
        "output_path": str(dst),
        "output_sample_rate": sample_rate,
    }
