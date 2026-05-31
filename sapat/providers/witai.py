# ABOUTME: Wit.ai transcription provider
# ABOUTME: Uses Wit.ai's /speech HTTP API for binary audio transcription

import json
import os
from pathlib import Path
from typing import Any, Optional

import requests

from sapat.providers import register
from sapat.providers.base import (
    AudioFormat,
    ProviderConfig,
    TranscriptionProvider,
    TranscriptionResult,
)


@register
class WitAIProvider(TranscriptionProvider):
    name = "witai"
    config = ProviderConfig(
        required_env_vars=["WITAI_ACCESS_TOKEN"],
        max_file_size_mb=25.0,
        preferred_format=AudioFormat.MP3,
        supports_correction=False,
        default_model="speech",
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
        token = os.getenv("WITAI_ACCESS_TOKEN", "")
        endpoint = os.getenv("WITAI_API_ENDPOINT", "https://api.wit.ai/speech")
        api_version = os.getenv("WITAI_API_VERSION", "20240304")
        content_type = os.getenv(
            "WITAI_CONTENT_TYPE",
            self._content_type(audio_file),
        )

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": content_type,
        }

        with open(audio_file, "rb") as f:
            audio_data = f.read()

        try:
            response = requests.post(
                endpoint,
                params={"v": api_version},
                headers=headers,
                data=audio_data,
                timeout=90,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Wit.ai transcription request failed: {exc}") from exc

        if response.status_code != 200:
            raise RuntimeError(
                f"Wit.ai transcription failed ({response.status_code}): "
                f"{response.text}"
            )

        payload = self._decode_response(response)
        text = self._extract_text(payload)
        return TranscriptionResult(text=text, raw_response=payload)

    def resolve_model(self, model_alias: str) -> str:
        # Wit.ai selects the speech model from the app configured for the token.
        return model_alias

    @staticmethod
    def _content_type(audio_file: str) -> str:
        extension = Path(audio_file).suffix.lower()
        content_types = {
            ".flac": "audio/flac",
            ".mp3": "audio/mpeg",
            ".ogg": "audio/ogg",
            ".wav": "audio/wav",
            ".webm": "audio/webm",
        }
        return content_types.get(extension, "application/octet-stream")

    @classmethod
    def _decode_response(cls, response) -> Any:
        try:
            return response.json()
        except ValueError:
            return cls._decode_json_stream(response.text)

    @staticmethod
    def _decode_json_stream(body: str) -> Any:
        text = body.strip()
        if not text:
            raise RuntimeError("Wit.ai response was empty.")

        decoder = json.JSONDecoder()
        values = []
        index = 0
        while index < len(text):
            while index < len(text) and text[index].isspace():
                index += 1
            if index >= len(text):
                break
            try:
                value, index = decoder.raw_decode(text, index)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Wit.ai response was not valid JSON: {exc}"
                ) from exc
            values.append(value)

        if not values:
            raise RuntimeError("Wit.ai response did not include a JSON payload.")
        return values[-1]

    @staticmethod
    def _extract_text(payload: Any) -> str:
        if isinstance(payload, dict):
            for key in ("text", "_text", "speech", "message"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        elif isinstance(payload, str) and payload.strip():
            return payload.strip()

        raise RuntimeError("Wit.ai response did not include transcript text.")
