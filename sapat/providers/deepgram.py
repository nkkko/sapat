# ABOUTME: Deepgram transcription provider
# ABOUTME: Uses Deepgram's pre-recorded Listen REST API

import os
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
class DeepgramProvider(TranscriptionProvider):
    name = "deepgram"
    config = ProviderConfig(
        required_env_vars=["DEEPGRAM_API_KEY"],
        max_file_size_mb=2000.0,
        preferred_format=AudioFormat.MP3,
        supports_correction=False,
        default_model=os.getenv("DEEPGRAM_MODEL", "nova-3"),
    )

    def resolve_model(self, model_alias: str) -> str:
        aliases = {
            "default": "nova-3",
            "nova": "nova-3",
            "enhanced": "enhanced",
            "base": "base",
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
        endpoint = os.getenv(
            "DEEPGRAM_API_ENDPOINT",
            "https://api.deepgram.com/v1/listen",
        )
        params = {
            "model": model,
            "language": language,
            "smart_format": os.getenv("DEEPGRAM_SMART_FORMAT", "true"),
        }

        if os.getenv("DEEPGRAM_DIARIZE", "").lower() in {"1", "true", "yes"}:
            params["diarize"] = "true"
        if os.getenv("DEEPGRAM_PUNCTUATE", "").lower() in {"1", "true", "yes"}:
            params["punctuate"] = "true"
        if prompt:
            params["keywords"] = prompt

        headers = {
            "Authorization": f"Token {os.getenv('DEEPGRAM_API_KEY', '')}",
            "Content-Type": "audio/mpeg",
        }

        with open(audio_file, "rb") as f:
            response = requests.post(
                endpoint,
                headers=headers,
                params=params,
                data=f.read(),
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"Deepgram transcription failed ({response.status_code}): "
                f"{response.text}"
            )

        payload = response.json()
        text = self._extract_transcript(payload)
        return TranscriptionResult(text=text, raw_response=payload)

    @staticmethod
    def _extract_transcript(payload: dict) -> str:
        try:
            alternatives = payload["results"]["channels"][0]["alternatives"]
            return alternatives[0].get("transcript", "")
        except (KeyError, IndexError, TypeError):
            return ""
