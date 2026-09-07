"""Deepgram prerecorded transcription using its binary-upload REST endpoint."""

import logging
import mimetypes
import os
from typing import Optional

import requests

from sapat.providers import register
from sapat.providers.base import (
    ProviderConfig,
    TranscriptionProvider,
    TranscriptionResult,
)

logger = logging.getLogger(__name__)


@register
class DeepgramProvider(TranscriptionProvider):
    name = "deepgram"
    config = ProviderConfig(
        required_env_vars=["DEEPGRAM_API_KEY"],
        default_model="nova-3",
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
        api_key = os.getenv("DEEPGRAM_API_KEY")
        if not api_key:
            raise ValueError("Set DEEPGRAM_API_KEY to use Deepgram")
        if prompt or temperature:
            logger.warning(
                "Deepgram does not use Sapat's transcription prompt or temperature; "
                "these options are ignored"
            )
        params = {"model": model, "smart_format": "true"}
        if language:
            params["language"] = language
        else:
            params["detect_language"] = "true"
        content_type = (
            mimetypes.guess_type(str(audio_file))[0] or "application/octet-stream"
        )
        with open(audio_file, "rb") as audio:
            response = requests.post(
                "https://api.deepgram.com/v1/listen",
                headers={
                    "Authorization": f"Token {api_key}",
                    "Content-Type": content_type,
                },
                params=params,
                data=audio,
                timeout=(10, 300),
            )
        if response.status_code != 200:
            # Do not echo response bodies, which may include submitted content.
            raise RuntimeError(
                f"Deepgram transcription failed (HTTP {response.status_code})"
            )
        try:
            result = response.json()
            channel = result["results"]["channels"][0]
            alternative = channel["alternatives"][0]
            text = alternative["transcript"]
            if not isinstance(text, str):
                raise TypeError("transcript must be a string")
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "Deepgram returned an invalid transcription response"
            ) from exc
        return TranscriptionResult(
            text=text,
            language=channel.get("detected_language") or language or None,
            duration=result.get("metadata", {}).get("duration"),
            raw_response=result,
        )
