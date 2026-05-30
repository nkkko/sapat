# ABOUTME: IBM Watson Speech to Text transcription provider
# ABOUTME: Uses the synchronous /v1/recognize REST API with API key auth

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
class IBMWatsonProvider(TranscriptionProvider):
    name = "ibm_watson"
    config = ProviderConfig(
        required_env_vars=["IBM_WATSON_STT_API_KEY", "IBM_WATSON_STT_URL"],
        max_file_size_mb=100.0,
        preferred_format=AudioFormat.WAV,
        supports_correction=False,
        default_model="en-US_BroadbandModel",
    )

    def resolve_model(self, model_alias: str) -> str:
        aliases = {
            "en": "en-US_BroadbandModel",
            "en-us": "en-US_BroadbandModel",
            "default": self.config.default_model,
        }
        return aliases.get(model_alias.lower(), model_alias)

    def transcribe(
        self,
        audio_file: str,
        model: str,
        language: str = "en",
        prompt: Optional[str] = None,
        temperature: float = 0,
        **kwargs,
    ) -> TranscriptionResult:
        api_key = os.getenv("IBM_WATSON_STT_API_KEY", "")
        service_url = os.getenv("IBM_WATSON_STT_URL", "").rstrip("/")
        timeout = float(os.getenv("IBM_WATSON_STT_TIMEOUT", "120"))

        params = {"model": model} if model else {}
        headers = {"Content-Type": self._content_type(audio_file)}

        with open(audio_file, "rb") as audio:
            response = requests.post(
                f"{service_url}/v1/recognize",
                auth=("apikey", api_key),
                headers=headers,
                params=params,
                data=audio,
                timeout=timeout,
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"IBM Watson transcription failed ({response.status_code}): {response.text}"
            )

        payload = response.json()
        text = self._extract_transcript(payload)
        return TranscriptionResult(text=text, raw_response=payload)

    @staticmethod
    def _content_type(audio_file: str) -> str:
        suffix = Path(audio_file).suffix.lower()
        return {
            ".flac": "audio/flac",
            ".mp3": "audio/mp3",
            ".mpeg": "audio/mpeg",
            ".ogg": "audio/ogg",
            ".wav": "audio/wav",
            ".webm": "audio/webm",
        }.get(suffix, "application/octet-stream")

    @staticmethod
    def _extract_transcript(payload: dict) -> str:
        transcripts = []
        for result in payload.get("results", []):
            alternatives = result.get("alternatives") or []
            if not alternatives:
                continue
            transcript = (alternatives[0].get("transcript") or "").strip()
            if transcript:
                transcripts.append(transcript)

        if not transcripts:
            raise RuntimeError("IBM Watson transcription returned no transcript text")

        return "\n".join(transcripts)
