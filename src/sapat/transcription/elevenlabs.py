import os
import mimetypes
from pathlib import Path

import requests
from dotenv import load_dotenv

from .base import TranscriptionBase


load_dotenv(".env")


SUPPORTED_EXTENSIONS = (".mp3", ".wav", ".flac", ".m4a", ".ogg")


class ElevenLabsTranscription(TranscriptionBase):
    """
    ElevenLabs Scribe implementation for transcription.
    """

    def __init__(self, temperature: float):
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        self.model = os.getenv("ELEVENLABS_MODEL", "scribe_v2")
        self.endpoint = os.getenv(
            "ELEVENLABS_API_ENDPOINT",
            "https://api.elevenlabs.io/v1/speech-to-text",
        )
        self.temperature = temperature

    def transcribe_audio(self, audio_file: str, **kwargs):
        """
        Transcribes an audio file using the ElevenLabs Speech to Text API.

        Parameters:
        - audio_file (str): Path to the audio file.
        - kwargs: Additional parameters for transcription.

        Returns:
        - dict: The transcription result.
        """
        self._validate_audio_file(audio_file)

        data = {
            "model_id": kwargs.get("model", self.model),
            "temperature": kwargs.get("temperature", self.temperature),
        }

        language = kwargs.get("language")
        if language:
            data["language_code"] = language

        headers = {
            "xi-api-key": self.api_key,
        }

        audio_path = Path(audio_file)
        content_type = mimetypes.guess_type(audio_path.name)[0] or "application/octet-stream"
        with audio_path.open("rb") as f:
            files = {
                "file": (audio_path.name, f, content_type),
            }
            response = requests.post(
                self.endpoint,
                headers=headers,
                data=data,
                files=files,
                timeout=120,
            )

        if response.status_code == 200:
            return response.json()

        raise Exception(
            f"Transcription failed with status {response.status_code}: {response.text}"
        )

    def _validate_audio_file(self, audio_file: str):
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY must be set in the environment.")

        if not os.path.exists(audio_file):
            raise ValueError(f"File {audio_file} does not exist.")

        extension = Path(audio_file).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported audio file format: {audio_file}. "
                f"Supported formats are {list(SUPPORTED_EXTENSIONS)}."
            )
