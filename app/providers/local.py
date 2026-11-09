"""Local voice provider — Stage 1 exposes configuration/capability status only.
No model weights are downloaded and no ML dependencies are installed."""

from pathlib import Path
from typing import List

from .. import config
from .base import (
    NotConfiguredError,
    NotImplementedYetError,
    Provider,
    ProviderCapabilities,
)


class LocalVoiceProvider(Provider):
    provider_id = "local"

    def missing_configuration(self) -> List[str]:
        missing = []
        if config.LOCAL_VOICE_MODEL_DIR is None:
            missing.append("LOCAL_VOICE_MODEL_DIR is not set")
        elif not Path(config.LOCAL_VOICE_MODEL_DIR).exists():
            missing.append(
                f"Model directory does not exist: {config.LOCAL_VOICE_MODEL_DIR}"
            )
        return missing

    def get_capabilities(self) -> ProviderCapabilities:
        configured = not self.missing_configuration()
        return ProviderCapabilities(
            name="Local Voice Engine",
            provider_id=self.provider_id,
            supports_cloud=False,
            supports_local=True,
            languages=["English", "Arabic"],
            configured=configured,
            generation_ready=False,  # Stage 1: no local inference
            notes=(
                "Model directory found — local generation will be enabled in a "
                "later stage." if configured
                else "Set LOCAL_VOICE_MODEL_DIR in .env and place model weights "
                "there to prepare for local generation."
            ),
        )

    def validate_voice(self, voice_profile: dict) -> bool:
        return bool(voice_profile) and bool(voice_profile.get("has_samples"))

    def generate(self, script: str, language: str, voice_profile: dict) -> dict:
        if self.missing_configuration():
            raise NotConfiguredError(
                "Local provider is not configured. Set LOCAL_VOICE_MODEL_DIR."
            )
        raise NotImplementedYetError(
            "Local generation is not implemented in Stage 1."
        )
