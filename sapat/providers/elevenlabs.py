# ABOUTME: ElevenLabs transcription provider
# ABOUTME: Uses ElevenLabs Scribe API with xi-api-key authentication for speech-to-text

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
class ElevenLabsProvider(TranscriptionProvider):
    name = "elevenlabs"
    config = ProviderConfig(
        required_env_vars=["ELEVENLABS_API_KEY"],
        default_model="scribe_v2",
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
        url = "https://api.elevenlabs.io/v1/speech-to-text"
        headers = {"xi-api-key": os.getenv("ELEVENLABS_API_KEY")}

        data = {
            "model_id": model,
            "temperature": temperature,
        }
        if language:
            data["language_code"] = language

        with open(audio_file, "rb") as f:
            response = requests.post(
                url,
                headers=headers,
                data=data,
                files={"file": f},
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"ElevenLabs transcription failed ({response.status_code}): {response.text}"
            )

        result = response.json()
        return TranscriptionResult(text=result.get("text", ""))
