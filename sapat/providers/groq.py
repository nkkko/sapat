# ABOUTME: Groq Cloud transcription provider
# ABOUTME: Uses Groq's Whisper API for fast speech-to-text

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
class GroqProvider(TranscriptionProvider):
    name = "groq"
    config = ProviderConfig(
        required_env_vars=["GROQ_API_KEY"],
        max_file_size_mb=25.0,
        preferred_format=AudioFormat.MP3,
        supports_correction=False,
        default_model="whisper-large-v3",
    )

    def resolve_model(self, model_alias: str) -> str:
        aliases = {
            "w": "whisper-large-v3",
            "dw": "distil-whisper-large-v3-en",
            "whisper": "whisper-large-v3",
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
        url = "https://api.groq.com/openai/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"}
        data = {
            "model": model,
            "response_format": "json",
            "temperature": temperature,
            "language": language,
        }
        if prompt:
            data["prompt"] = prompt

        with open(audio_file, "rb") as f:
            files = {"file": f}
            response = requests.post(url, headers=headers, data=data, files=files)

        if response.status_code != 200:
            raise RuntimeError(f"Groq transcription failed ({response.status_code}): {response.text}")

        result = response.json()
        return TranscriptionResult(text=result.get("text", ""))
