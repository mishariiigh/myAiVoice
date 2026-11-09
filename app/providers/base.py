"""Common provider interface for all voice engines.

Every engine (cloud or local) implements this interface, so the rest of
the application never depends on a specific vendor.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class ProviderCapabilities:
    """What a provider can do right now."""

    name: str
    provider_id: str
    supports_cloud: bool
    supports_local: bool
    languages: List[str] = field(default_factory=lambda: ["English", "Arabic"])
    configured: bool = False          # are all credentials/resources present?
    generation_ready: bool = False    # is actual generation implemented?
    notes: str = ""


class ProviderError(Exception):
    """Base error for provider failures."""


class NotConfiguredError(ProviderError):
    """Provider lacks credentials or resources."""


class NotImplementedYetError(ProviderError):
    """Generation is not implemented in this stage."""


class Provider:
    """Abstract interface every voice engine must implement."""

    provider_id: str = "abstract"

    def get_capabilities(self) -> ProviderCapabilities:
        raise NotImplementedError

    def validate_voice(self, voice_profile: dict) -> bool:
        """Check whether the given local voice profile is usable by this engine."""
        raise NotImplementedError

    def generate(self, script: str, language: str, voice_profile: dict) -> dict:
        """Produce speech. Returns {'status': ..., 'audio_path': ...} on success."""
        raise NotImplementedError

    # Optional hook subclasses may use to report missing configuration pieces.
    def missing_configuration(self) -> List[str]:
        return []
