# ABOUTME: Azure AI Speech REST transcription provider
# ABOUTME: Sends 16 kHz mono WAV audio to the short-audio speech-to-text API

import os
from pathlib import Path
from typing import Optional

import requests

from sapat.providers import register
from sapat.providers.base import (AudioFormat, ProviderConfig,
                                  TranscriptionProvider, TranscriptionResult)

LANGUAGE_ALIASES = {
    "bg": "bg-BG",
    "de": "de-DE",
    "en": "en-US",
    "es": "es-ES",
    "fr": "fr-FR",
    "it": "it-IT",
    "ja": "ja-JP",
    "pt": "pt-BR",
    "ru": "ru-RU",
    "zh": "zh-CN",
}

RECOGNITION_MODES = {
    "conversation": "conversation",
    "default": "conversation",
    "dictation": "dictation",
    "interactive": "interactive",
}

RESPONSE_FORMATS = {"simple", "detailed"}
PROFANITY_MODES = {"masked", "removed", "raw"}


@register
class AzureSpeechProvider(TranscriptionProvider):
    """Azure AI Speech short-audio REST transcription provider."""

    name = "azure_speech"
    config = ProviderConfig(
        required_env_vars=["AZURE_SPEECH_KEY"],
        max_file_size_mb=1.8,
        preferred_format=AudioFormat.WAV,
        supports_correction=False,
        default_model="conversation",
    )

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("AZURE_SPEECH_KEY", "")
        self.region = os.getenv("AZURE_SPEECH_REGION", "")
        self.endpoint = os.getenv("AZURE_SPEECH_ENDPOINT", "")
        self.response_format = os.getenv(
            "AZURE_SPEECH_RESPONSE_FORMAT", "simple"
        ).lower()
        self.profanity = os.getenv("AZURE_SPEECH_PROFANITY")

    @classmethod
    def is_available(cls) -> bool:
        return bool(
            os.getenv("AZURE_SPEECH_KEY")
            and (os.getenv("AZURE_SPEECH_ENDPOINT") or os.getenv("AZURE_SPEECH_REGION"))
        )

    def resolve_model(self, model_alias: str) -> str:
        return RECOGNITION_MODES.get(model_alias, model_alias)

    def transcribe(
        self,
        audio_file: str,
        model: str,
        language: str = "en",
        prompt: Optional[str] = None,
        temperature: float = 0,
        **kwargs,
    ) -> TranscriptionResult:
        recognition_mode = self.resolve_model(model or self.config.default_model)
        url = self._build_endpoint(recognition_mode)
        params = self._build_params(language)
        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Content-Type": self._content_type(audio_file),
            "Accept": "application/json",
        }

        with open(audio_file, "rb") as f:
            response = requests.post(
                url,
                headers=headers,
                params=params,
                data=f.read(),
            )

        if response.status_code != 200:
            raise RuntimeError(
                "Azure AI Speech transcription failed "
                f"({response.status_code}): {self._error_text(response)}"
            )

        payload = response.json()
        return self._parse_result(payload, params["language"])

    def _build_endpoint(self, recognition_mode: str) -> str:
        if recognition_mode not in RECOGNITION_MODES.values():
            raise ValueError(
                "Azure AI Speech recognition mode must be one of: "
                "conversation, dictation, interactive."
            )

        configured = self.endpoint.strip().rstrip("/")
        if configured:
            lowered = configured.lower()
            if lowered.endswith("/cognitiveservices/v1"):
                return configured
            if "/speech/recognition/" in lowered:
                return f"{configured}/cognitiveservices/v1"

            if "cognitiveservices.azure.com" in lowered:
                path_prefix = "stt/speech/recognition"
            else:
                path_prefix = "speech/recognition"
            return f"{configured}/{path_prefix}/{recognition_mode}/cognitiveservices/v1"

        region = self.region.strip()
        if not region:
            raise ValueError(
                "Set AZURE_SPEECH_ENDPOINT or AZURE_SPEECH_REGION before using "
                "azure_speech."
            )
        return (
            f"https://{region}.stt.speech.microsoft.com/"
            f"speech/recognition/{recognition_mode}/cognitiveservices/v1"
        )

    def _build_params(self, language: str) -> dict:
        response_format = self.response_format or "simple"
        if response_format not in RESPONSE_FORMATS:
            raise ValueError(
                "AZURE_SPEECH_RESPONSE_FORMAT must be 'simple' or 'detailed'."
            )

        params = {
            "language": self._normalize_language(language),
            "format": response_format,
        }

        if self.profanity:
            profanity = self.profanity.lower()
            if profanity not in PROFANITY_MODES:
                raise ValueError(
                    "AZURE_SPEECH_PROFANITY must be 'masked', 'removed', or 'raw'."
                )
            params["profanity"] = profanity

        return params

    def _parse_result(self, payload: dict, language: str) -> TranscriptionResult:
        status = payload.get("RecognitionStatus")
        if status == "NoMatch":
            return TranscriptionResult(text="", language=language, raw_response=payload)
        if status and status != "Success":
            raise RuntimeError(
                f"Azure AI Speech recognition failed with status '{status}'."
            )

        text = self._extract_text(payload)
        return TranscriptionResult(
            text=text,
            language=language,
            duration=self._duration_seconds(payload),
            raw_response=payload,
        )

    @staticmethod
    def _extract_text(payload: dict) -> str:
        if payload.get("DisplayText"):
            return payload["DisplayText"]

        for candidate in payload.get("NBest", []):
            for key in ("Display", "Lexical", "ITN", "MaskedITN"):
                value = candidate.get(key)
                if value:
                    return value
        return ""

    @staticmethod
    def _duration_seconds(payload: dict) -> Optional[float]:
        duration = payload.get("Duration")
        try:
            return int(duration) / 10_000_000 if duration is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_language(language: str) -> str:
        if not language:
            return "en-US"
        return LANGUAGE_ALIASES.get(language.lower(), language)

    @staticmethod
    def _content_type(audio_file: str) -> str:
        extension = Path(audio_file).suffix.lower()
        if extension == ".ogg":
            return "audio/ogg; codecs=opus"
        return "audio/wav; codecs=audio/pcm; samplerate=16000"

    @staticmethod
    def _error_text(response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return response.text

        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict) and error.get("message"):
                return error["message"]
            if payload.get("message"):
                return payload["message"]
        return response.text
