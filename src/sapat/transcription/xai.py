import mimetypes
import os

import requests
from dotenv import load_dotenv

from .base import TranscriptionBase

# Load environment variables
load_dotenv(".env")


class XAITranscription(TranscriptionBase):
    """
    xAI Speech-to-Text API implementation for transcription.
    """

    def __init__(self, temperature: float):
        """
        Initializes the XAITranscription class.

        Parameters:
        - temperature (float): Accepted for CLI compatibility with other providers.
        """
        self.api_key = os.getenv("XAI_API_KEY")
        self.endpoint = os.getenv("XAI_STT_API_ENDPOINT", "https://api.x.ai/v1/stt")
        self.temperature = temperature
        self.max_file_size_mb = 500

    def transcribe_audio(self, audio_file: str, **kwargs):
        """
        Transcribes an audio file using xAI Speech-to-Text.

        Parameters:
        - audio_file (str): Path to the audio file.
        - kwargs: Additional parameters for transcription.

        Returns:
        - dict: The transcription result from xAI.
        """
        self._validate_configuration()
        self._validate_audio_file(audio_file)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        data = self._build_data(**kwargs)

        with open(audio_file, "rb") as f:
            files = {
                "file": (
                    os.path.basename(audio_file),
                    f,
                    mimetypes.guess_type(audio_file)[0] or "application/octet-stream",
                )
            }
            response = requests.post(
                self.endpoint,
                headers=headers,
                data=data,
                files=files,
            )

        if 200 <= response.status_code < 300:
            return response.json()
        raise Exception(f"Transcription failed: {response.text}")

    def _validate_configuration(self):
        if not self.api_key:
            raise ValueError("XAI_API_KEY is required for xAI transcription.")

    def _build_data(self, **kwargs):
        data = [("format", self._bool_value(os.getenv("XAI_STT_FORMAT", "true")))]

        language = kwargs.get("language") or os.getenv("XAI_STT_LANGUAGE")
        if language:
            data.append(("language", language))

        for env_name, field_name in [
            ("XAI_STT_DIARIZE", "diarize"),
            ("XAI_STT_MULTICHANNEL", "multichannel"),
            ("XAI_STT_FILLER_WORDS", "filler_words"),
        ]:
            value = os.getenv(env_name)
            if value:
                data.append((field_name, self._bool_value(value)))

        for env_name, field_name in [
            ("XAI_STT_CHANNELS", "channels"),
            ("XAI_STT_AUDIO_FORMAT", "audio_format"),
            ("XAI_STT_SAMPLE_RATE", "sample_rate"),
        ]:
            value = os.getenv(env_name)
            if value:
                data.append((field_name, value))

        for keyterm in self._keyterms(kwargs.get("keyterms")):
            data.append(("keyterm", keyterm))

        return data

    @staticmethod
    def _bool_value(value):
        return "true" if str(value).strip().lower() in {"1", "true", "yes", "on"} else "false"

    @staticmethod
    def _keyterms(value=None):
        keyterms = value
        if keyterms is None:
            keyterms = os.getenv("XAI_STT_KEYTERMS", "")

        if isinstance(keyterms, str):
            parts = keyterms.split(",")
        else:
            parts = list(keyterms)

        return [part.strip() for part in parts if part and part.strip()]

    def _validate_audio_file(self, audio_file: str):
        """
        Validates the audio file for size and format.

        Parameters:
        - audio_file (str): Path to the audio file.

        Raises:
        - Exception: If the file is invalid.
        """
        if not os.path.exists(audio_file):
            raise ValueError(f"File {audio_file} does not exist.")

        file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            raise Exception(f"File size exceeds the maximum limit of {self.max_file_size_mb} MB.")

        valid_extensions = [
            ".mp3",
            ".wav",
            ".flac",
            ".ogg",
            ".opus",
            ".aac",
            ".mp4",
            ".m4a",
            ".mkv",
        ]
        if not any(str(audio_file).endswith(ext) for ext in valid_extensions):
            raise ValueError(
                f"Unsupported audio file format: {audio_file}. Supported formats are {valid_extensions}."
            )
