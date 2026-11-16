"""ElevenLabs cloud provider — Stage 3A-1: real TTS integration.

Implemented here:
- Dynamic API-key retrieval via config.get_elevenlabs_api_key() (always
  reflects the current environment; the key is never logged or embedded in
  exceptions).
- Voice selection through the profile's ElevenLabs voice ID mapping
  (voice_profiles.elevenlabs_voice_id, schema v3).
- Text-to-speech with the `eleven_multilingual_v2` model (English + Arabic
  from a single model; no language_code parameter is sent — that model
  auto-detects the language).
- MP3 response handling (bytes written into local storage).
- Voice-list retrieval with pagination (GET /v1/voices?page_size=...).
- Typed error handling: missing credentials, authentication failures,
  rate limits, timeouts, and invalid/unexpected responses.

Intentionally NOT implemented (Stage 3A-2 and beyond): voice cloning and
speech-to-speech.

Security notes:
- The API key is sent only in the `xi-api-key` header and never appears in
  exception messages, logs, or capability notes.
- Network failures are mapped to typed ProviderError subclasses so the UI
  can present a safe, generic message.
"""

import os
from typing import Dict, List, Optional, Tuple

import requests

from .. import config
from .base import (
    NotConfiguredError,
    NotImplementedYetError,
    Provider,
    ProviderAuthenticationError,
    ProviderCapabilities,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderVoiceMappingError,
)

# eleven_multilingual_v2 covers both supported languages and produces the
# highest-quality output; per-language voice IDs are not required.
TTS_MODEL_ID = "eleven_multilingual_v2"
DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"

# Do not send language_code for this model (auto-detects English/Arabic).
SUPPORTED_LANGUAGES = ("English", "Arabic")

DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_PAGE_SIZE = 100  # API maximum for /v1/voices


class ElevenLabsProvider(Provider):
    provider_id = "elevenlabs"

    # --- Configuration -------------------------------------------------------------

    def missing_configuration(self) -> List[str]:
        missing = []
        if not config.get_elevenlabs_api_key():
            missing.append("ELEVENLABS_API_KEY is not set")
        return missing

    def get_capabilities(self) -> ProviderCapabilities:
        configured = not self.missing_configuration()
        return ProviderCapabilities(
            name="ElevenLabs (Cloud)",
            provider_id=self.provider_id,
            supports_cloud=True,
            supports_local=False,
            languages=list(SUPPORTED_LANGUAGES),
            configured=configured,
            generation_ready=configured,  # TTS is implemented in Stage 3A-1
            notes=(
                "Cloud text-to-speech via eleven_multilingual_v2 (English and "
                "Arabic). Voice identity comes from the profile's mapped "
                "ElevenLabs voice ID."
                if configured
                else "Set ELEVENLABS_API_KEY in .env to enable cloud generation."
            ),
        )

    def validate_voice(self, voice_profile: dict) -> bool:
        """A profile is usable for cloud TTS when it has a mapped voice ID."""
        return bool(voice_profile) and bool(
            str(voice_profile.get("elevenlabs_voice_id") or "").strip()
        )

    # --- Internal helpers ------------------------------------------------------------

    @staticmethod
    def _headers() -> Dict[str, str]:
        key = config.get_elevenlabs_api_key()
        return {
            "xi-api-key": key,
            "Accept": "application/json",
        }

    @staticmethod
    def _require_configured() -> None:
        if not config.get_elevenlabs_api_key():
            raise NotConfiguredError(
                "ElevenLabs is not configured. Add ELEVENLABS_API_KEY to .env."
            )

    def _resolve_voice_id(self, voice_profile: dict) -> str:
        voice_id = str((voice_profile or {}).get("elevenlabs_voice_id") or "").strip()
        if not voice_id:
            raise ProviderVoiceMappingError(
                "This voice profile has no ElevenLabs voice ID mapped. "
                "Map the profile to an ElevenLabs voice before generating."
            )
        return voice_id

    @staticmethod
    def _check_response_status(response: requests.Response) -> None:
        """Map HTTP status codes onto typed provider errors.

        Exception messages never include the API key or request headers.
        """
        status = response.status_code
        if 200 <= status < 300:
            return
        if status in (401, 403):
            raise ProviderAuthenticationError(
                "ElevenLabs rejected the credentials. Check that "
                "ELEVENLABS_API_KEY is valid and has not been revoked."
            )
        if status == 429:
            retry_after: Optional[float] = None
            raw = response.headers.get("retry-after")
            if raw:
                try:
                    retry_after = float(raw)
                except ValueError:
                    retry_after = None
            hint = (
                f" Retry after {int(retry_after)} seconds."
                if retry_after is not None
                else ""
            )
            raise ProviderRateLimitError(
                "ElevenLabs rate limit reached." + hint,
                retry_after_seconds=retry_after,
            )
        if 400 <= status < 500:
            raise ProviderRequestError(
                f"ElevenLabs rejected the request (HTTP {status})."
            )
        raise ProviderResponseError(
            f"ElevenLabs returned an unexpected status (HTTP {status})."
        )

    # --- TTS generation ----------------------------------------------------------------

    def generate(self, script: str, language: str, voice_profile: dict,
                 emotion: str = "Neutral",
                 expression_reference_id: str = None) -> dict:
        """Synthesize `script` with the profile's mapped voice.

        Returns {'status': 'completed', 'audio_bytes': bytes,
                 'audio_format': 'mp3', 'model_id': ...} on success.
        Raises a typed ProviderError subclass on any failure.
        """
        self._require_configured()
        voice_id = self._resolve_voice_id(voice_profile)
        text = (script or "").strip()
        if not text:
            raise ProviderRequestError("Cannot generate speech from an empty script.")

        url = (
            f"{config.get_elevenlabs_base_url().rstrip('/')}"
            f"/v1/text-to-speech/{voice_id}"
        )
        payload = {
            "text": text,
            "model_id": TTS_MODEL_ID,
            # No language_code: eleven_multilingual_v2 auto-detects language.
        }
        try:
            response = requests.post(
                url,
                json=payload,
                headers={**self._headers(), "Accept": "audio/mpeg"},
                params={"output_format": DEFAULT_OUTPUT_FORMAT},
                timeout=DEFAULT_TIMEOUT_SECONDS,
            )
        except requests.Timeout as exc:
            raise ProviderTimeoutError(
                "ElevenLabs did not respond in time. Try again."
            ) from exc
        except requests.RequestException as exc:
            # Network/DNS/body errors — message is deliberately generic.
            raise ProviderResponseError(
                "Could not reach the ElevenLabs service."
            ) from exc

        self._check_response_status(response)

        audio = response.content
        content_type = response.headers.get("Content-Type", "")
        if not audio:
            raise ProviderResponseError(
                "ElevenLabs returned an empty audio response."
            )
        if "mpeg" not in content_type and "octet-stream" not in content_type:
            raise ProviderResponseError(
                "ElevenLabs returned an unexpected content type for audio."
            )

        return {
            "status": "completed",
            "audio_bytes": audio,
            "audio_format": "mp3",
            "model_id": TTS_MODEL_ID,
            "voice_id": voice_id,
        }

    # --- Voice list with pagination ------------------------------------------------------

    def list_voices(self) -> List[Dict[str, object]]:
        """Fetch the account's voices, following cursor pagination.

        Returns a list of dicts with keys:
          - voice_id: str
          - name: str
          - is_owner: bool  (True for cloned/owned voices, False for premade)
          - category: str  (e.g. 'premade', 'cloned', or '')
        Sorted by name.
        """
        self._require_configured()
        base = config.get_elevenlabs_base_url().rstrip("/")
        voices: List[Dict[str, object]] = []
        cursor: Optional[str] = None
        seen_pages = 0
        max_pages = 50  # hard stop against runaway pagination

        while seen_pages < max_pages:
            params = {"page_size": DEFAULT_PAGE_SIZE}
            if cursor:
                params["cursor"] = cursor
            try:
                response = requests.get(
                    f"{base}/v1/voices",
                    headers=self._headers(),
                    params=params,
                    timeout=DEFAULT_TIMEOUT_SECONDS,
                )
            except requests.Timeout as exc:
                raise ProviderTimeoutError(
                    "ElevenLabs did not respond in time while listing voices."
                ) from exc
            except requests.RequestException as exc:
                raise ProviderResponseError(
                    "Could not reach the ElevenLabs service."
                ) from exc

            self._check_response_status(response)

            try:
                body = response.json()
            except ValueError as exc:
                raise ProviderResponseError(
                    "ElevenLabs returned a malformed voice list."
                ) from exc

            page = body.get("voices") if isinstance(body, dict) else None
            if not isinstance(page, list):
                raise ProviderResponseError(
                    "ElevenLabs returned a malformed voice list."
                )
            for item in page:
                if not isinstance(item, dict):
                    continue
                voice_id = item.get("voice_id")
                name = item.get("name")
                if voice_id and name:
                    voices.append({
                        "voice_id": voice_id,
                        "name": name,
                        "is_owner": bool(item.get("is_owner")),
                        "category": item.get("category") or "",
                    })

            cursor = body.get("next_page_token") or body.get("cursor")
            has_more = body.get("has_more")
            if not cursor or (has_more is False and not cursor):
                break
            seen_pages += 1

        voices.sort(key=lambda v: str(v["name"]).lower())
        return voices
