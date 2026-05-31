# ABOUTME: OpenAI transcription provider
# ABOUTME: Uses OpenAI's Audio API for file-based speech-to-text

import os
from typing import Optional

import requests

from sapat.providers import register
from sapat.providers.base import (
    AudioFormat,
    ProviderConfig,
    TranscriptionResult,
)
from sapat.providers.openai_compat import OpenAICompatProvider


DEFAULT_ENDPOINT = "https://api.openai.com/v1/audio/transcriptions"
DIARIZATION_MODEL = "gpt-4o-transcribe-diarize"


@register
class OpenAIProvider(OpenAICompatProvider):
    name = "openai"
    config = ProviderConfig(
        required_env_vars=["OPENAI_API_KEY"],
        max_file_size_mb=25.0,
        preferred_format=AudioFormat.MP3,
        supports_correction=False,
        default_model=os.getenv(
            "OPENAI_TRANSCRIPTION_MODEL",
            os.getenv("OPENAI_MODEL", "gpt-4o-transcribe"),
        ),
    )
    _env_key_for_auth = "OPENAI_API_KEY"

    @property
    def base_url(self) -> str:
        return (
            os.getenv("OPENAI_TRANSCRIPTION_ENDPOINT")
            or os.getenv("OPENAI_API_ENDPOINT")
            or DEFAULT_ENDPOINT
        )

    def resolve_model(self, model_alias: str) -> str:
        aliases = {
            "4o": "gpt-4o-transcribe",
            "gpt4o": "gpt-4o-transcribe",
            "mini": "gpt-4o-mini-transcribe",
            "4o-mini": "gpt-4o-mini-transcribe",
            "gpt4o-mini": "gpt-4o-mini-transcribe",
            "w": "whisper-1",
            "whisper": "whisper-1",
            "diarize": DIARIZATION_MODEL,
            "diarization": DIARIZATION_MODEL,
        }
        return aliases.get(model_alias, model_alias)

    def _build_data(
        self, model: str, language: str, prompt, temperature: float, **kwargs
    ) -> dict:
        data = super()._build_data(model, language, prompt, temperature, **kwargs)

        response_format = kwargs.get("response_format")
        if response_format:
            data["response_format"] = response_format

        if model == DIARIZATION_MODEL:
            data["response_format"] = response_format or "diarized_json"
            data["chunking_strategy"] = kwargs.get("chunking_strategy", "auto")

            speaker_names = kwargs.get("known_speaker_names")
            if speaker_names:
                data["known_speaker_names[]"] = speaker_names

            speaker_references = kwargs.get("known_speaker_references")
            if speaker_references:
                data["known_speaker_references[]"] = speaker_references

        include = kwargs.get("include")
        if include:
            data["include[]"] = include

        timestamp_granularities = kwargs.get("timestamp_granularities")
        if timestamp_granularities:
            data["timestamp_granularities[]"] = timestamp_granularities

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
        model = self.resolve_model(model)
        if model == DIARIZATION_MODEL and prompt:
            raise ValueError("OpenAI diarization transcriptions do not support prompts")
        if model == DIARIZATION_MODEL:
            unsupported = [
                name
                for name in ("include", "timestamp_granularities")
                if kwargs.get(name)
            ]
            if unsupported:
                unsupported_list = ", ".join(unsupported)
                raise ValueError(
                    f"OpenAI diarization transcriptions do not support: {unsupported_list}"
                )

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

        if data.get("response_format") == "text":
            return TranscriptionResult(text=response.text, raw_response=response.text)

        result = response.json()
        return TranscriptionResult(
            text=result.get("text", ""),
            language=result.get("language"),
            duration=result.get("duration"),
            segments=result.get("segments"),
            raw_response=result,
        )
