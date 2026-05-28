# ABOUTME: NVIDIA NIM transcription provider
# ABOUTME: Uses NVIDIA's ASR NIM API with OpenAI-compatible endpoint

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
class NvidiaProvider(TranscriptionProvider):
    name = "nvidia"
    config = ProviderConfig(
        required_env_vars=["NVIDIA_API_KEY"],
        max_file_size_mb=25.0,
        preferred_format=AudioFormat.MP3,
        supports_correction=False,
        default_model="nvidia/parakeet-ctc-0.6b-asr",
    )

    def resolve_model(self, model_alias: str) -> str:
        aliases = {
            "parakeet": "nvidia/parakeet-ctc-0.6b-asr",
            "canary": "nvidia/canary-1b",
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
        api_key = os.getenv("NVIDIA_API_KEY", "")
        base_url = os.getenv(
            "NVIDIA_BASE_URL",
            "https://integrate.api.nvidia.com/v1",
        ).rstrip("/")
        url = f"{base_url}/audio/transcriptions"

        headers = {"Authorization": f"Bearer {api_key}"}
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
            response = requests.post(
                url, headers=headers, data=data, files=files
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"NVIDIA transcription failed ({response.status_code}): "
                f"{response.text}"
            )

        result = response.json()
        return TranscriptionResult(text=result.get("text", ""))
