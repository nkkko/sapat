import mimetypes
import os

import requests
from dotenv import load_dotenv

from .base import TranscriptionBase


load_dotenv(".env")


class TogetherAITranscription(TranscriptionBase):
    """
    Together AI API implementation for transcription.
    """

    def __init__(self, temperature: float, response_format: str = "json"):
        self.api_key = os.getenv("TOGETHER_API_KEY")
        self.model = os.getenv("TOGETHER_MODEL", "openai/whisper-large-v3")
        self.endpoint = os.getenv(
            "TOGETHER_API_ENDPOINT",
            "https://api.together.xyz/v1/audio/transcriptions",
        )
        self.temperature = temperature
        self.response_format = response_format
        self.max_file_size_mb = 100

    def transcribe_audio(self, audio_file: str, **kwargs):
        """
        Transcribes an audio file using Together AI speech-to-text.
        """
        self._validate_config()
        self._validate_audio_file(audio_file)

        data = {
            "model": kwargs.get("model", self.model),
            "response_format": kwargs.get("response_format", self.response_format),
            "temperature": kwargs.get("temperature", self.temperature),
        }

        if kwargs.get("language"):
            data["language"] = kwargs["language"]
        if kwargs.get("prompt"):
            data["prompt"] = kwargs["prompt"]

        content_type = mimetypes.guess_type(audio_file)[0] or "audio/mpeg"

        with open(audio_file, "rb") as audio:
            files = {
                "file": (os.path.basename(audio_file), audio, content_type),
            }
            response = requests.post(
                self.endpoint,
                headers={"Authorization": f"Bearer {self.api_key}"},
                data=data,
                files=files,
            )

        if response.status_code != 200:
            raise Exception(f"Transcription failed: {response.text}")

        return self._extract_transcript(response.json())

    def _validate_config(self):
        if not self.api_key:
            raise ValueError("TOGETHER_API_KEY is required for Together AI transcription.")

    def _validate_audio_file(self, audio_file: str):
        if not os.path.exists(audio_file):
            raise ValueError(f"File {audio_file} does not exist.")

        file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            raise Exception(f"File size exceeds the maximum limit of {self.max_file_size_mb} MB.")

        valid_extensions = [".mp3", ".wav", ".m4a", ".webm", ".flac", ".ogg", ".opus", ".aac"]
        if not any(str(audio_file).endswith(ext) for ext in valid_extensions):
            raise ValueError(f"Unsupported audio file format: {audio_file}. Supported formats are {valid_extensions}.")

    @staticmethod
    def _extract_transcript(payload):
        if isinstance(payload, dict):
            return payload.get("text", "")
        return ""
