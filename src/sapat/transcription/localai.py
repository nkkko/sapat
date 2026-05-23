import os

import requests

from .base import TranscriptionBase

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv:
    load_dotenv(".env")


class LocalAITranscription(TranscriptionBase):
    """
    LocalAI OpenAI-compatible transcription implementation.
    """

    def __init__(self, temperature: float, response_format: str = "json"):
        self.base_url = os.getenv("LOCALAI_BASE_URL")
        self.endpoint = os.getenv("LOCALAI_API_ENDPOINT") or self._build_endpoint(self.base_url)
        self.api_key = os.getenv("LOCALAI_API_KEY")
        self.model = os.getenv("LOCALAI_MODEL", "whisper-1")
        self.temperature = temperature
        self.response_format = response_format
        self.max_file_size_mb = 25

    def transcribe_audio(self, audio_file: str, **kwargs):
        """
        Transcribes an audio file using LocalAI's OpenAI-compatible endpoint.
        """
        self._validate_configuration()
        self._validate_audio_file(audio_file)

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = {
            "model": kwargs.get("model", self.model),
            "response_format": kwargs.get("response_format", self.response_format),
            "temperature": kwargs.get("temperature", self.temperature),
        }

        if "language" in kwargs:
            data["language"] = kwargs["language"]
        if "prompt" in kwargs:
            data["prompt"] = kwargs["prompt"]

        with open(audio_file, "rb") as f:
            response = requests.post(
                self.endpoint,
                headers=headers,
                data=data,
                files={"file": f},
            )

        if response.status_code == 200:
            if data["response_format"] in ["json", "verbose_json"]:
                return response.json()
            return response.text

        raise Exception(f"Transcription failed: {response.text}")

    @staticmethod
    def _build_endpoint(base_url):
        if not base_url:
            return None
        return f"{base_url.rstrip('/')}/v1/audio/transcriptions"

    def _validate_configuration(self):
        if not self.endpoint:
            raise ValueError(
                "LOCALAI_BASE_URL or LOCALAI_API_ENDPOINT must be set for LocalAI transcription."
            )

    def _validate_audio_file(self, audio_file: str):
        if not os.path.exists(audio_file):
            raise ValueError(f"File {audio_file} does not exist.")

        file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            raise Exception(
                f"File size exceeds the maximum limit of {self.max_file_size_mb} MB."
            )

        valid_extensions = [".mp3", ".wav", ".flac"]
        if not any(str(audio_file).endswith(ext) for ext in valid_extensions):
            raise ValueError(
                f"Unsupported audio file format: {audio_file}. "
                f"Supported formats are {valid_extensions}."
            )
