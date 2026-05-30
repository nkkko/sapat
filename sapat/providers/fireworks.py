# ABOUTME: Fireworks AI transcription provider
# ABOUTME: Uses Fireworks Audio API directly without a provider SDK dependency

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
class FireworksProvider(TranscriptionProvider):
    name = "fireworks"
    config = ProviderConfig(
        required_env_vars=["FIREWORKS_API_KEY"],
        max_file_size_mb=1000.0,
        preferred_format=AudioFormat.WAV,
        supports_correction=False,
        default_model="whisper-v3",
    )

    def resolve_model(self, model_alias: str) -> str:
        aliases = {
            "default": "whisper-v3",
            "whisper": "whisper-v3",
            "turbo": "whisper-v3-turbo",
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
        headers = {"Authorization": os.getenv("FIREWORKS_API_KEY", "")}
        data = {
            "model": model,
            "response_format": "json",
            "temperature": temperature,
        }
        if language:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt

        with open(audio_file, "rb") as f:
            response = requests.post(
                self._endpoint_for_model(model),
                headers=headers,
                data=data,
                files={"file": f},
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"Fireworks transcription failed ({response.status_code}): {response.text}"
            )

        payload = response.json()
        return TranscriptionResult(
            text=self._extract_text(payload),
            language=payload.get("language") if isinstance(payload, dict) else None,
            raw_response=payload,
        )

    @staticmethod
    def _endpoint_for_model(model: str) -> str:
        if model == "whisper-v3-turbo":
            host = "https://audio-turbo.api.fireworks.ai"
        else:
            host = "https://audio-prod.api.fireworks.ai"
        return f"{host}/v1/audio/transcriptions"

    @staticmethod
    def _extract_text(payload) -> str:
        if isinstance(payload, str):
            text = payload
        elif isinstance(payload, dict):
            text = payload.get("text", "")
        else:
            text = ""

        if not text:
            raise RuntimeError("Fireworks response contained no transcript text")
        return text
