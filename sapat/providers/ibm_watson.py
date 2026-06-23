# ABOUTME: IBM Watson Speech to Text transcription provider
# ABOUTME: Uses the Watson Speech to Text REST recognize endpoint

import os
from pathlib import Path
from typing import Optional

import requests

from sapat.providers import register
from sapat.providers.base import (
    AudioFormat,
    ProviderConfig,
    TranscriptionProvider,
    TranscriptionResult,
)


@register
class IBMWatsonProvider(TranscriptionProvider):
    name = "ibm_watson"
    config = ProviderConfig(
        required_env_vars=["IBM_WATSON_STT_API_KEY", "IBM_WATSON_STT_URL"],
        max_file_size_mb=100.0,
        preferred_format=AudioFormat.MP3,
        supports_correction=False,
        default_model="en-US_BroadbandModel",
    )

    def resolve_model(self, model_alias: str) -> str:
        aliases = {
            "default": "en-US_BroadbandModel",
            "en": "en-US_BroadbandModel",
            "en-us": "en-US_BroadbandModel",
            "en-gb": "en-GB_BroadbandModel",
            "es": "es-ES_BroadbandModel",
            "fr": "fr-FR_BroadbandModel",
            "de": "de-DE_BroadbandModel",
            "ja": "ja-JP_BroadbandModel",
            "pt": "pt-BR_BroadbandModel",
        }
        return aliases.get(model_alias.lower(), model_alias)

    def transcribe(
        self,
        audio_file: str,
        model: str,
        language: str = "en",
        prompt: Optional[str] = None,
        temperature: float = 0,
        **kwargs,
    ) -> TranscriptionResult:
        api_key = os.getenv("IBM_WATSON_STT_API_KEY", "")
        service_url = os.getenv("IBM_WATSON_STT_URL", "").rstrip("/")
        url = f"{service_url}/v1/recognize"

        params = {
            "model": self.resolve_model(model or self.config.default_model),
        }
        content_type = kwargs.get("content_type") or self._content_type(audio_file)
        headers = {"Content-Type": content_type}

        with open(audio_file, "rb") as f:
            response = requests.post(
                url,
                auth=("apikey", api_key),
                headers=headers,
                params=params,
                data=f.read(),
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"IBM Watson transcription failed ({response.status_code}): "
                f"{response.text}"
            )

        payload = response.json()
        return TranscriptionResult(
            text=self._extract_text(payload),
            language=language,
            raw_response=payload,
        )

    @staticmethod
    def _extract_text(payload: dict) -> str:
        transcripts = []
        for result in payload.get("results", []):
            alternatives = result.get("alternatives") or []
            if alternatives:
                transcript = alternatives[0].get("transcript", "").strip()
                if transcript:
                    transcripts.append(transcript)
        return " ".join(transcripts)

    @staticmethod
    def _content_type(audio_file: str) -> str:
        extension = Path(audio_file).suffix.lower()
        content_types = {
            ".flac": "audio/flac",
            ".mp3": "audio/mp3",
            ".mpeg": "audio/mpeg",
            ".ogg": "audio/ogg",
            ".wav": "audio/wav",
            ".webm": "audio/webm",
        }
        return content_types.get(extension, "application/octet-stream")
