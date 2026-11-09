"""Provider registry: the application resolves engines by id, never by class."""

from typing import Dict, List, Optional

from .base import Provider


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: Dict[str, Provider] = {}

    def register(self, provider: Provider) -> None:
        if provider.provider_id in self._providers:
            raise ValueError(f"Provider '{provider.provider_id}' already registered.")
        self._providers[provider.provider_id] = provider

    def unregister(self, provider_id: str) -> None:
        self._providers.pop(provider_id, None)

    def get(self, provider_id: str) -> Optional[Provider]:
        return self._providers.get(provider_id)

    def list_providers(self) -> List[Provider]:
        return list(self._providers.values())

    def ids(self) -> List[str]:
        return list(self._providers.keys())


_default_registry: Optional[ProviderRegistry] = None


def get_registry() -> ProviderRegistry:
    """Build (once) the default registry with the Stage 1 providers."""
    global _default_registry
    if _default_registry is None:
        from .elevenlabs import ElevenLabsProvider
        from .local import LocalVoiceProvider

        reg = ProviderRegistry()
        reg.register(ElevenLabsProvider())
        reg.register(LocalVoiceProvider())
        _default_registry = reg
    return _default_registry
