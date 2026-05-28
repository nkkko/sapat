# ABOUTME: LocalAI self-hosted transcription provider
# ABOUTME: Uses LocalAI's OpenAI-compatible endpoint with configurable base URL

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
class LocalAIProvider(TranscriptionProvider):
    name = "localai"
    config = ProviderConfig(
        required_env_vars=["LOCALAI_BASE_URL"],
        default_model="whisper-1",
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
        base_url = os.getenv("LOCALAI_BASE_URL", "").rstrip("/")
        url = f"{base_url}/v1/audio/transcriptions"

        headers = {}
        api_key = os.getenv("LOCALAI_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        data = {
            "model": model,
            "response_format": "json",
            "temperature": temperature,
            "language": language,
        }
        if prompt:
            data["prompt"] = prompt

        with open(audio_file, "rb") as f:
            response = requests.post(
                url,
                headers=headers,
                data=data,
                files={"file": f},
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"LocalAI transcription failed ({response.status_code}): {response.text}"
            )

        result = response.json()
        return TranscriptionResult(text=result.get("text", ""))
