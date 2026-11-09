"""Central configuration for Mishari Voice Studio.

All settings come from environment variables (optionally via a local .env
file). No secrets are ever hardcoded here, and API keys are never printed
to logs or the UI.
"""

import os
from pathlib import Path

try:  # Optional: load a local .env file if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - graceful degradation
    pass


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


BASE_DIR = Path(__file__).resolve().parent.parent

APP_NAME = "Mishari Voice Studio"
APP_VERSION = "0.1.0 (Stage 1)"

# --- Storage -----------------------------------------------------------------
DATA_DIR = Path(_env("DATA_DIR", str(BASE_DIR / "data"))).resolve()
VOICES_DIR = DATA_DIR / "voices"
SAMPLES_DIR = DATA_DIR / "samples"
GENERATED_DIR = DATA_DIR / "generated"
DB_PATH = DATA_DIR / _env("DATABASE_NAME", "mishari_voice_studio.db")

MAX_UPLOAD_MB = int(_env("MAX_UPLOAD_MB", "25"))
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a"}
AUDIO_MIME_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/mpeg",
    "audio/mp4",
    "audio/x-m4a",
    "audio/mp4a-latm",
    "application/octet-stream",  # browsers often send this for audio
}

SUPPORTED_LANGUAGES = ("English", "Arabic")

# --- Providers ---------------------------------------------------------------
ELEVENLABS_API_KEY = _env("ELEVENLABS_API_KEY")
ELEVENLABS_BASE_URL = _env("ELEVENLABS_BASE_URL", "https://api.elevenlabs.io")
LOCAL_VOICE_MODEL_DIR = Path(_env("LOCAL_VOICE_MODEL_DIR", "")) if _env("LOCAL_VOICE_MODEL_DIR") else None


def ensure_directories() -> None:
    """Create the storage tree if missing (idempotent, safe to call anytime)."""
    for d in (DATA_DIR, VOICES_DIR, SAMPLES_DIR, GENERATED_DIR):
        d.mkdir(parents=True, exist_ok=True)
