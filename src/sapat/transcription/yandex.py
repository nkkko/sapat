import os
import subprocess
import tempfile
from pathlib import Path

import requests
from dotenv import load_dotenv

from .base import TranscriptionBase


load_dotenv(".env")


class YandexSpeechKitTranscription(TranscriptionBase):
    """
    Yandex SpeechKit API implementation for short audio transcription.
    """

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

    def __init__(self, temperature: float, response_format: str = "json"):
        """
        Initializes the YandexSpeechKitTranscription class.

        Parameters:
        - temperature (float): Accepted for CLI compatibility. SpeechKit v1 does not use it.
        - response_format (str): Accepted for CLI compatibility. SpeechKit v1 returns JSON.
        """
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
        self.temperature = temperature
        self.response_format = response_format
        self.max_file_size_mb = float(os.getenv("YANDEX_MAX_FILE_SIZE_MB", "1"))

    def transcribe_audio(self, audio_file: str, **kwargs):
        """
        Transcribes an audio file using Yandex SpeechKit synchronous recognition.

        Parameters:
        - audio_file (str): Path to the audio file.
        - kwargs: Additional parameters for transcription.

        Returns:
        - dict: The transcription result with a text field compatible with Sapat.
        """
        self._validate_audio_file(audio_file)
        headers = self._build_headers()
        params = self._build_params(kwargs)

        prepared_file = self._convert_for_yandex(audio_file)
        try:
            self._validate_prepared_file(prepared_file)
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
            return {"text": payload.get("result", ""), "result": payload.get("result", "")}

        raise Exception(f"Yandex SpeechKit transcription failed: {response.text}")

    def _build_headers(self):
        if self.api_key:
            authorization = f"Api-Key {self.api_key}"
        elif self.iam_token:
            authorization = f"Bearer {self.iam_token}"
        else:
            raise ValueError("Set YANDEX_API_KEY or YANDEX_IAM_TOKEN before using --api yandex.")

        return {
            "Authorization": authorization,
            "Content-Type": "application/octet-stream",
        }

    def _build_params(self, kwargs):
        language = self._normalize_language(kwargs.get("language", self.language))
        params = {
            "lang": language,
            "topic": kwargs.get("topic", self.topic),
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

    def _normalize_language(self, language):
        if not language:
            return self.language
        return self.LANGUAGE_ALIASES.get(str(language).lower(), language)

    def _convert_for_yandex(self, audio_file):
        if self.audio_format not in {"oggopus", "lpcm"}:
            raise ValueError("YANDEX_AUDIO_FORMAT must be 'oggopus' or 'lpcm'.")

        suffix = ".ogg" if self.audio_format == "oggopus" else ".lpcm"
        handle, prepared_file = tempfile.mkstemp(prefix="sapat-yandex-", suffix=suffix)
        os.close(handle)

        if self.audio_format == "oggopus":
            command = [
                "ffmpeg",
                "-y",
                "-i",
                audio_file,
                "-vn",
                "-ac",
                "1",
                "-ar",
                "48000",
                "-c:a",
                "libopus",
                "-b:a",
                "24k",
                prepared_file,
            ]
        else:
            command = [
                "ffmpeg",
                "-y",
                "-i",
                audio_file,
                "-vn",
                "-ac",
                "1",
                "-ar",
                self.sample_rate_hertz,
                "-f",
                "s16le",
                "-acodec",
                "pcm_s16le",
                prepared_file,
            ]

        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            self._cleanup_prepared_file(prepared_file)
            raise Exception("Failed to convert audio for Yandex SpeechKit.") from exc

        return prepared_file

    def _validate_prepared_file(self, audio_file):
        file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            raise Exception(
                f"Yandex SpeechKit synchronous recognition supports files up to "
                f"{self.max_file_size_mb} MB. Use a shorter clip or an async provider."
            )

    def _cleanup_prepared_file(self, audio_file):
        try:
            Path(audio_file).unlink(missing_ok=True)
        except TypeError:
            path = Path(audio_file)
            if path.exists():
                path.unlink()

    def _validate_audio_file(self, audio_file: str):
        if not os.path.exists(audio_file):
            raise ValueError(f"File {audio_file} does not exist.")

        valid_extensions = [
            ".mp3",
            ".wav",
            ".flac",
            ".ogg",
            ".opus",
            ".m4a",
            ".aac",
            ".mp4",
            ".webm",
            ".mov",
        ]
        if not any(str(audio_file).lower().endswith(ext) for ext in valid_extensions):
            raise ValueError(f"Unsupported audio file format: {audio_file}. Supported formats are {valid_extensions}.")
