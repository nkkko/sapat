# ABOUTME: Sarvam AI transcription provider
# ABOUTME: Uses Sarvam's Saaras speech-to-text REST API with BCP-47 language codes

import os
from typing import Dict, Optional

import requests

from sapat.providers import register
from sapat.providers.base import (
    AudioFormat,
    ProviderConfig,
    TranscriptionProvider,
    TranscriptionResult,
)

LANGUAGE_ALIASES: Dict[str, str] = {
    "as": "as-IN",
    "bn": "bn-IN",
    "brx": "brx-IN",
    "doi": "doi-IN",
    "en": "en-IN",
    "gu": "gu-IN",
    "hi": "hi-IN",
    "kn": "kn-IN",
    "kok": "kok-IN",
    "ks": "ks-IN",
    "mai": "mai-IN",
    "ml": "ml-IN",
    "mni": "mni-IN",
    "mr": "mr-IN",
    "ne": "ne-IN",
    "od": "od-IN",
    "pa": "pa-IN",
    "sa": "sa-IN",
    "sat": "sat-IN",
    "sd": "sd-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "ur": "ur-IN",
}


@register
class SarvamProvider(TranscriptionProvider):
    name = "sarvam"
    config = ProviderConfig(
        required_env_vars=["SARVAM_API_KEY"],
        max_file_size_mb=25.0,
        preferred_format=AudioFormat.MP3,
        supports_correction=False,
        default_model="saaras:v3",
    )

    def resolve_model(self, model_alias: str) -> str:
        aliases = {
            "v2": "saaras:v2",
            "v3": "saaras:v3",
        }
        return aliases.get(model_alias, model_alias)

    def transcribe(
        self,
        audio_file: str,
        model: str,
        language: str = "en",
        prompt: Optional[str] = None,
        temperature: float = 0,
        **kwargs,
    ) -> TranscriptionResult:
        api_key = os.getenv("SARVAM_API_KEY", "")
        endpoint = os.getenv(
            "SARVAM_STT_ENDPOINT",
            "https://api.sarvam.ai/speech-to-text",
        )

        data = {
            "model": model,
            "mode": kwargs.get("mode", "transcribe"),
        }

        language_code = self._resolve_language_code(language)
        if language_code:
            data["language_code"] = language_code

        headers = {"api-subscription-key": api_key}

        with open(audio_file, "rb") as f:
            files = {"file": f}
            response = requests.post(
                endpoint, headers=headers, data=data, files=files
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"Sarvam transcription failed ({response.status_code}): "
                f"{response.text}"
            )

        payload = response.json()
        text = payload.get("transcript", "")
        return TranscriptionResult(
            text=text,
            raw_response=payload,
        )

    def _resolve_language_code(self, language: Optional[str]) -> Optional[str]:
        if not language:
            return None

        normalized = language.strip().lower()
        if not normalized or normalized == "auto":
            return None
        if normalized == "unknown":
            return "unknown"
        if "-" in language:
            return language
        return LANGUAGE_ALIASES.get(normalized, language)
