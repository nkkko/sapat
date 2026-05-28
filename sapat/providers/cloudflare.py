# ABOUTME: Cloudflare Workers AI transcription provider
# ABOUTME: Uses Cloudflare's Whisper model via the Workers AI REST API

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
class CloudflareProvider(TranscriptionProvider):
    name = "cloudflare"
    config = ProviderConfig(
        required_env_vars=["CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN"],
        max_file_size_mb=25.0,
        preferred_format=AudioFormat.MP3,
        supports_correction=False,
        default_model="@cf/openai/whisper",
    )

    def resolve_model(self, model_alias: str) -> str:
        aliases = {
            "whisper": "@cf/openai/whisper",
            "default": "@cf/openai/whisper",
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
        account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
        api_token = os.getenv("CLOUDFLARE_API_TOKEN", "")

        url = (
            f"https://api.cloudflare.com/client/v4/accounts/"
            f"{account_id}/ai/run/{model}"
        )
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": self._content_type(audio_file),
        }

        with open(audio_file, "rb") as f:
            response = requests.post(url, headers=headers, data=f.read())

        if response.status_code != 200:
            raise RuntimeError(
                f"Cloudflare transcription failed ({response.status_code}): "
                f"{response.text}"
            )

        payload = response.json()
        # Cloudflare wraps results in {"success": true, "result": {...}}
        if isinstance(payload, dict) and "result" in payload:
            result = payload["result"]
        else:
            result = payload

        text = result.get("text", "") if isinstance(result, dict) else str(result)
        return TranscriptionResult(text=text)

    @staticmethod
    def _content_type(audio_file: str) -> str:
        extension = Path(audio_file).suffix.lower()
        content_types = {
            ".flac": "audio/flac",
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
        }
        return content_types.get(extension, "application/octet-stream")
