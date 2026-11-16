"""Local voice provider — real Audar-TTS-V1-Flash inference via subprocess.

The heavy ML stack (llama-cpp-python, torch, neucodec) lives in the isolated
`.venv-local` environment; this provider talks to it through
`app.engines.audar_local.run_local_generation`, a JSON-line subprocess
bridge. Nothing here falls back to any cloud provider.
"""
import os
import sqlite3
from pathlib import Path
from typing import List, Optional

from .. import config, db
from ..engines import audar_local
from .base import (
    NotConfiguredError,
    Provider,
    ProviderCapabilities,
    ProviderError,
)


class ReferenceNotReadyError(ProviderError):
    """The profile has no sample with a verified transcript + valid audio."""


class LocalVoiceProvider(Provider):
    provider_id = "local"

    def missing_configuration(self) -> List[str]:
        missing = []
        if not audar_local.engine_ready():
            missing.extend(audar_local.missing_pieces())
        return missing

    def get_capabilities(self) -> ProviderCapabilities:
        problems = self.missing_configuration()
        configured = not problems
        if configured:
            notes = ("Audar-TTS-V1-Flash (GGUF, CPU) ready — voice cloning "
                     "runs fully offline.")
        else:
            notes = "Not ready: " + "; ".join(problems)
        return ProviderCapabilities(
            name="Local Voice Engine",
            provider_id=self.provider_id,
            supports_cloud=False,
            supports_local=True,
            languages=["English", "Arabic"],
            configured=configured,
            generation_ready=configured,
            notes=notes,
        )

    def validate_voice(self, voice_profile: dict) -> bool:
        if not voice_profile or not voice_profile.get("id"):
            return False
        return db.get_usable_reference_sample(voice_profile["id"]) is not None

    def generate(self, script: str, language: str,
                 voice_profile: dict, emotion: str = "Neutral",
                 expression_reference_id: Optional[str] = None) -> dict:
        text = (script or "").strip()
        if not text:
            raise ProviderError("The script is empty. Enter some text to generate.")

        problems = self.missing_configuration()
        if problems:
            raise NotConfiguredError(
                "Local voice engine is not ready: " + "; ".join(problems))

        profile_id = (voice_profile or {}).get("id")
        if not profile_id:
            raise ProviderError("No voice profile selected.")

        sample = db.get_usable_reference_sample(profile_id)
        if sample is None:
            raise ReferenceNotReadyError(
                "This profile has no usable voice reference. Upload or record "
                "a sample and enter its exact transcript in the Voice Library "
                "first. The transcript is required — it is never invented.")

        result = audar_local.run_local_generation(
            text=text,
            ref_wav_path=sample["file_path"],
            ref_text=(sample["transcript"] or "").strip(),
        )
        wav_path = Path(result["wav_path"])
        audio_bytes = wav_path.read_bytes()
        try:
            wav_path.unlink(missing_ok=True)
        except OSError:
            pass
        return {
            "audio_bytes": audio_bytes,
            "audio_format": "wav",
            "model_id": "Audar-TTS-V1-Flash-Q4_K_M",
            "duration_seconds": result.get("duration_seconds"),
            "sample_rate": result.get("sample_rate"),
        }
