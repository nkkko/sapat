# ABOUTME: Yandex SpeechKit synchronous transcription provider
# ABOUTME: Converts audio to OggOpus, then sends to Yandex STT recognize endpoint

import os
import subprocess
import tempfile
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

LANGUAGE_ALIASES = {
    "en": "en-US",
    "ru": "ru-RU",
    "kk": "kk-KZ",
    "uz": "uz-UZ",
    "tr": "tr-TR",
    "de": "de-DE",
    "fr": "fr-FR",
    "es": "es-ES",
    "it": "it-IT",
}


@register
class YandexProvider(TranscriptionProvider):
    """Yandex SpeechKit synchronous recognition provider."""

    name = "yandex"
    config = ProviderConfig(
        required_env_vars=[],
        preferred_format=AudioFormat.OGG,
        default_model="general",
    )

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("YANDEX_API_KEY")
        self.iam_token = os.getenv("YANDEX_IAM_TOKEN")
        self.folder_id = os.getenv("YANDEX_FOLDER_ID")
        self.endpoint = os.getenv(
            "YANDEX_API_ENDPOINT",
            "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize",
        )
        self.language = os.getenv("YANDEX_LANGUAGE", "en-US")
        self.topic = os.getenv("YANDEX_TOPIC", "general")
        self.audio_format = os.getenv("YANDEX_AUDIO_FORMAT", "oggopus").lower()
        self.sample_rate_hertz = os.getenv("YANDEX_SAMPLE_RATE_HERTZ", "48000")
        self.profanity_filter = os.getenv("YANDEX_PROFANITY_FILTER")
        self.raw_results = os.getenv("YANDEX_RAW_RESULTS")
        self.max_file_size_mb = float(os.getenv("YANDEX_MAX_FILE_SIZE_MB", "1"))

    @classmethod
    def is_available(cls) -> bool:
        has_api_key = bool(os.getenv("YANDEX_API_KEY"))
        has_iam = bool(os.getenv("YANDEX_IAM_TOKEN"))
        return has_api_key or has_iam

    def transcribe(
        self,
        audio_file: str,
        model: str,
        language: str = "en",
        prompt: Optional[str] = None,
        temperature: float = 0,
        **kwargs,
    ) -> TranscriptionResult:
        headers = self._build_headers()
        params = self._build_params(language)

        prepared_file = self._convert_for_yandex(audio_file)
        try:
            with open(prepared_file, "rb") as f:
                response = requests.post(
                    self.endpoint,
                    headers=headers,
                    params=params,
                    data=f,
                )
        finally:
            self._cleanup_prepared_file(prepared_file)

        if response.status_code == 200:
            payload = response.json()
            return TranscriptionResult(text=payload.get("result", ""))

        raise RuntimeError(
            f"Yandex SpeechKit transcription failed: {response.text}"
        )

    def _build_headers(self) -> dict:
        if self.api_key:
            authorization = f"Api-Key {self.api_key}"
        elif self.iam_token:
            authorization = f"Bearer {self.iam_token}"
        else:
            raise ValueError(
                "Set YANDEX_API_KEY or YANDEX_IAM_TOKEN before using the yandex provider."
            )

        return {
            "Authorization": authorization,
            "Content-Type": "application/octet-stream",
        }

    def _build_params(self, language: str) -> dict:
        normalized = self._normalize_language(language)
        params = {
            "lang": normalized,
            "topic": self.topic,
            "format": self.audio_format,
        }

        if self.audio_format == "lpcm":
            params["sampleRateHertz"] = self.sample_rate_hertz
        if self.folder_id and not self.api_key:
            params["folderId"] = self.folder_id
        if self.profanity_filter is not None:
            params["profanityFilter"] = self.profanity_filter
        if self.raw_results is not None:
            params["rawResults"] = self.raw_results

        return params

    def _normalize_language(self, language: str) -> str:
        if not language:
            return self.language
        return LANGUAGE_ALIASES.get(language.lower(), language)

    def _convert_for_yandex(self, audio_file: str) -> str:
        if self.audio_format not in {"oggopus", "lpcm"}:
            raise ValueError("YANDEX_AUDIO_FORMAT must be 'oggopus' or 'lpcm'.")

        suffix = ".ogg" if self.audio_format == "oggopus" else ".lpcm"
        handle, prepared_file = tempfile.mkstemp(
            prefix="sapat-yandex-", suffix=suffix
        )
        os.close(handle)

        if self.audio_format == "oggopus":
            command = [
                "ffmpeg", "-y", "-i", audio_file,
                "-vn", "-ac", "1", "-ar", "48000",
                "-c:a", "libopus", "-b:a", "24k",
                prepared_file,
            ]
        else:
            command = [
                "ffmpeg", "-y", "-i", audio_file,
                "-vn", "-ac", "1", "-ar", self.sample_rate_hertz,
                "-f", "s16le", "-acodec", "pcm_s16le",
                prepared_file,
            ]

        try:
            subprocess.run(
                command, check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as exc:
            self._cleanup_prepared_file(prepared_file)
            raise RuntimeError(
                "Failed to convert audio for Yandex SpeechKit."
            ) from exc

        return prepared_file

    def _cleanup_prepared_file(self, audio_file: str) -> None:
        try:
            Path(audio_file).unlink(missing_ok=True)
        except TypeError:
            path = Path(audio_file)
            if path.exists():
                path.unlink()
