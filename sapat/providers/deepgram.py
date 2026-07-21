# ABOUTME: Deepgram transcription provider
# ABOUTME: Uses Deepgram's pre-recorded audio transcription endpoint

import os
from typing import Optional

import requests

from sapat.providers import register
from sapat.providers.base import ProviderConfig, TranscriptionProvider, TranscriptionResult


@register
class DeepgramProvider(TranscriptionProvider):
    """Deepgram pre-recorded transcription provider."""

    name = "deepgram"
    config = ProviderConfig(
        required_env_vars=["DEEPGRAM_API_KEY"],
        default_model="nova-3",
    )

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("DEEPGRAM_API_KEY", "")
        self.endpoint = os.getenv(
            "DEEPGRAM_API_ENDPOINT",
            "https://api.deepgram.com/v1/listen",
        )

    def transcribe(
        self,
        audio_file: str,
        model: str,
        language: str = "en",
        prompt: Optional[str] = None,
        temperature: float = 0,
        **kwargs,
    ) -> TranscriptionResult:
        params = {
            "model": model,
            "smart_format": "true",
        }
        if language:
            params["language"] = language

        headers = {"Authorization": f"Token {self.api_key}"}

        with open(audio_file, "rb") as f:
            response = requests.post(
                self.endpoint,
                headers=headers,
                params=params,
                data=f,
                timeout=120,
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"Deepgram transcription failed ({response.status_code}): {response.text}"
            )

        payload = response.json()
        alternative = (
            payload.get("results", {})
            .get("channels", [{}])[0]
            .get("alternatives", [{}])[0]
        )
        return TranscriptionResult(
            text=alternative.get("transcript", ""),
            language=payload.get("metadata", {}).get("language"),
            raw_response=payload,
        )
