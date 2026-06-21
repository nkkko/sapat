# ABOUTME: OpenRouter speech-to-text provider
# ABOUTME: Uses OpenRouter's dedicated JSON/base64 transcription endpoint

import base64
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
class OpenRouterProvider(TranscriptionProvider):
    name = "openrouter"
    endpoint = "https://openrouter.ai/api/v1/audio/transcriptions"
    supported_formats = {"wav", "mp3", "flac", "m4a", "ogg", "webm", "aac"}
    config = ProviderConfig(
        required_env_vars=["OPENROUTER_API_KEY"],
        max_file_size_mb=25.0,
        preferred_format=AudioFormat.MP3,
        supports_correction=False,
        default_model="openai/whisper-large-v3",
    )

    def resolve_model(self, model_alias: str) -> str:
        aliases = {
            "whisper": "openai/whisper-large-v3",
            "whisper-large": "openai/whisper-large-v3",
            "whisper-turbo": "openai/whisper-large-v3-turbo",
            "gpt-4o": "openai/gpt-4o-transcribe",
            "gpt-4o-mini": "openai/gpt-4o-mini-transcribe",
            "voxtral-mini": "mistralai/voxtral-mini-transcribe",
            "qwen-flash": "qwen/qwen3-asr-flash-2026-02-10",
            "parakeet": "nvidia/parakeet-tdt-0.6b-v3",
            "mai": "microsoft/mai-transcribe-1.5",
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
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json",
        }
        payload = self._build_payload(audio_file, model, language, temperature)

        response = requests.post(self.endpoint, headers=headers, json=payload)

        if response.status_code != 200:
            raise RuntimeError(
                f"OpenRouter transcription failed ({response.status_code}): "
                f"{response.text}"
            )

        result = response.json()
        usage = result.get("usage", {}) if isinstance(result, dict) else {}
        duration = usage.get("seconds") if isinstance(usage, dict) else None
        return TranscriptionResult(
            text=result.get("text", ""),
            duration=duration if isinstance(duration, (int, float)) else None,
            raw_response=result,
        )

    def _build_payload(
        self,
        audio_file: str,
        model: str,
        language: str,
        temperature: float,
    ) -> dict:
        with open(audio_file, "rb") as f:
            audio_data = base64.b64encode(f.read()).decode("ascii")

        payload = {
            "model": model,
            "input_audio": {
                "data": audio_data,
                "format": self._audio_format(audio_file),
            },
            "temperature": temperature,
        }
        if language and language != "auto":
            payload["language"] = language
        return payload

    def _audio_format(self, audio_file: str) -> str:
        suffix = Path(audio_file).suffix.lower().lstrip(".")
        if suffix in self.supported_formats:
            return suffix
        return self.config.preferred_format.value
