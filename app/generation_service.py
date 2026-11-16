"""Generation workflow orchestration — Stage 3A-2.

Connects the approved Stage 3A-1 pieces into one consistent workflow:

    validate inputs -> provider.generate() -> create 'processing' record ->
    store_generated_audio() -> add_audio_file() -> mark 'completed'

Consistency strategy:
- The generation row is created ONLY after the provider returned audio, and
  only ever marked 'completed' AFTER the audio file exists on disk AND the
  audio_files row exists. A crash between steps therefore never leaves a
  'completed' record without audio.
- If persisting the audio or the DB rows fails after the provider call, the
  function cleans up the orphan file / audio row, marks the generation
  'failed', and re-raises a secret-safe ProviderError for the UI.
- All error messages surfaced to the UI are pre-sanitized; API keys and
  internal details never appear.

This module is UI-free so it can be tested without Streamlit.
"""

from pathlib import Path
from typing import Optional

from . import config, db, storage
from .providers.base import (
    NotConfiguredError,
    ProviderError,
    ProviderRateLimitError,
)
from .providers.elevenlabs import ElevenLabsProvider


def build_generation_error(exc: Exception) -> str:
    """Translate a provider/storage exception into a secret-safe UI message."""
    if isinstance(exc, NotConfiguredError):
        return ("ElevenLabs is not configured. Add ELEVENLABS_API_KEY in "
                "Settings (.env) and try again.")
    if isinstance(exc, ProviderRateLimitError):
        retry = getattr(exc, "retry_after_seconds", None)
        suffix = f" Try again in about {int(retry)} seconds." if retry else ""
        return f"The provider rate limit was reached.{suffix}"
    if isinstance(exc, ProviderError):
        text = str(exc)
        # Belt-and-braces: never let any credential-looking substring through.
        key = config.get_elevenlabs_api_key()
        if key and key in text:
            return "Generation failed due to a provider error."
        return text
    # Unexpected exception (storage IO, DB, ...): generic message only.
    return "Generation failed while saving the audio. Please try again."


def run_generation(provider: ElevenLabsProvider, script: str, language: str,
                   voice_profile: dict, emotion: str = "Neutral",
                   expression_reference_id: Optional[str] = None) -> dict:
    """Execute one full generation and persist it consistently.

    Returns the completed generation row (joined view) on success.
    Raises ProviderError (secret-safe) on any failure.
    """
    text = (script or "").strip()
    if not text:
        raise ProviderError("The script is empty. Enter some text to generate.")

    profile = dict(voice_profile or {})
    if not profile.get("id"):
        raise ProviderError("Select a voice profile before generating.")

    # Ask the provider first: if this fails, nothing has been persisted.
    result = provider.generate(text, language, profile,
                               emotion=emotion,
                               expression_reference_id=expression_reference_id)
    audio_bytes = result.get("audio_bytes")
    audio_format = result.get("audio_format") or "mp3"
    if not audio_bytes:
        raise ProviderError("The provider returned no audio. Please try again.")

    gid = db.add_generation(
        text, language, provider.provider_id,
        voice_profile_id=profile.get("id"),
        provider_model=result.get("model_id"),
        status="processing",
        emotion=emotion or None,
        expression_reference_id=expression_reference_id or None,
    )

    stored_path: Optional[str] = None
    audio_file_id: Optional[str] = None
    try:
        stored_path = storage.store_generated_audio(audio_bytes, audio_format, gid)
        audio_file_id = db.add_audio_file(
            stored_path, audio_format, len(audio_bytes)
        )
        if not audio_file_id:
            raise RuntimeError("audio_files insert returned no id")
        db.update_generation(gid, status="completed", audio_file_id=audio_file_id)
    except Exception as exc:
        # Consistent rollback: remove the orphan audio row, the orphan file,
        # and mark the generation failed — never mask the original error.
        cleanup_error_message = build_generation_error(exc)
        try:
            if audio_file_id:
                db.delete_audio_file(audio_file_id)
        except Exception:
            pass
        try:
            if stored_path:
                storage.delete_generated_file(stored_path)
        except Exception:
            pass
        try:
            db.update_generation(gid, status="failed",
                                 error_message=cleanup_error_message)
        except Exception:
            pass
        if isinstance(exc, ProviderError):
            raise
        raise ProviderError(cleanup_error_message) from exc

    completed = next((r for r in db.list_generations() if r["id"] == gid), None)
    if completed is None:
        raise ProviderError("Generation record could not be finalized.")
    return dict(completed)


def resolve_safe_audio_path(audio_path: Optional[str]) -> Optional[Path]:
    """Return the audio path only if it exists AND is inside GENERATED_DIR.

    Guards the history page against deleted files and unsafe paths. Returns
    None when the path is missing, outside GENERATED_DIR, or None.
    """
    if not audio_path:
        return None
    resolved = Path(audio_path).resolve()
    generated_root = Path(config.GENERATED_DIR).resolve()
    if generated_root not in resolved.parents and resolved != generated_root:
        return None
    if not resolved.is_file():
        return None
    return resolved
