"""ElevenLabs cloud provider — Stage 1 exposes configuration/capability status
only. Actual API generation is intentionally NOT implemented yet."""

from typing import List

from .. import config
from .base import (
    NotConfiguredError,
    NotImplementedYetError,
    Provider,
    ProviderCapabilities,
)


class ElevenLabsProvider(Provider):
    provider_id = "elevenlabs"

    def missing_configuration(self) -> List[str]:
        missing = []
        if not config.ELEVENLABS_API_KEY:
            missing.append("ELEVENLABS_API_KEY is not set")
        return missing

    def get_capabilities(self) -> ProviderCapabilities:
        configured = not self.missing_configuration()
        return ProviderCapabilities(
            name="ElevenLabs (Cloud)",
            provider_id=self.provider_id,
            supports_cloud=True,
            supports_local=False,
            languages=["English", "Arabic"],
            configured=configured,
            generation_ready=False,  # Stage 1: no actual generation
            notes=(
                "API key detected — cloud generation will be enabled in a later "
                "stage." if configured
                else "Set ELEVENLABS_API_KEY in .env to enable cloud generation."
            ),
        )

    def validate_voice(self, voice_profile: dict) -> bool:
        # Stage 1: a profile needs at least one sample to be cloud-clonable later.
        return bool(voice_profile) and bool(voice_profile.get("has_samples"))

    def generate(self, script: str, language: str, voice_profile: dict) -> dict:
        if self.missing_configuration():
            raise NotConfiguredError(
                "ElevenLabs is not configured. Add ELEVENLABS_API_KEY to .env."
            )
        # Even when configured, Stage 1 does not call the API.
        raise NotImplementedYetError(
            "Cloud generation is not implemented in Stage 1."
        )
