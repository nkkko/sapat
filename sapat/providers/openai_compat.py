# ABOUTME: Shared mixin for OpenAI-compatible transcription providers
# ABOUTME: Handles the common multipart POST to /audio/transcriptions pattern

import os
from typing import Optional

import requests

from sapat.providers.base import (
    AudioFormat,
    ProviderConfig,
    TranscriptionProvider,
    TranscriptionResult,
)


class OpenAICompatProvider(TranscriptionProvider):
    """
    Mixin for providers following the OpenAI /audio/transcriptions pattern.

    Subclasses set: base_url, _env_key_for_auth
    And optionally: _auth_header_name, _auth_header_prefix
    """

    base_url: str = ""
    _auth_header_name: str = "Authorization"
    _auth_header_prefix: str = "Bearer "
    _env_key_for_auth: str = ""

    def _get_auth_value(self) -> str:
        return f"{self._auth_header_prefix}{os.getenv(self._env_key_for_auth, '')}"

    def _build_data(self, model: str, language: str, prompt, temperature: float, **kwargs) -> dict:
        data = {
            "model": model,
            "response_format": "json",
            "temperature": temperature,
            "language": language,
        }
        if prompt:
            data["prompt"] = prompt
        return data

    def transcribe(
        self,
        audio_file: str,
        model: str,
        language: str = "en",
        prompt: Optional[str] = None,
        temperature: float = 0,
        **kwargs,
    ) -> TranscriptionResult:
        headers = {self._auth_header_name: self._get_auth_value()}
        data = self._build_data(model, language, prompt, temperature, **kwargs)

        with open(audio_file, "rb") as f:
            response = requests.post(
                self.base_url,
                headers=headers,
                data=data,
                files={"file": f},
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"{self.name} transcription failed ({response.status_code}): {response.text}"
            )

        result = response.json()
        return TranscriptionResult(text=result.get("text", ""))
