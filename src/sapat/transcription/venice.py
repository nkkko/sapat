import os

import requests
from dotenv import load_dotenv

from .base import TranscriptionBase

load_dotenv(".env")


class VeniceTranscription(TranscriptionBase):
    """
    Venice API implementation for transcription.
    """

    def __init__(self, temperature: float, response_format: str = "json"):
        """
        Initializes the VeniceTranscription class.

        Parameters:
        - temperature (float): Kept for CLI parity with other providers.
        - response_format (str): Default response format for transcription.
        """
        self.api_key = os.getenv("VENICE_API_KEY")
        self.model = os.getenv("VENICE_MODEL", "openai/whisper-large-v3")
        self.endpoint = os.getenv(
            "VENICE_API_ENDPOINT",
            "https://api.venice.ai/api/v1/audio/transcriptions",
        )
        self.timestamps = os.getenv("VENICE_TIMESTAMPS", "false").lower()
        self.temperature = temperature
        self.response_format = response_format

    def transcribe_audio(self, audio_file: str, **kwargs):
        """
        Transcribes an audio file using Venice's transcription API.

        Parameters:
        - audio_file (str): Path to the audio file.
        - kwargs: Additional parameters for transcription.

        Returns:
        - dict or str: The transcription result.
        """
        self._validate_audio_file(audio_file)
        if not self.api_key:
            raise ValueError("VENICE_API_KEY must be set to use the Venice API.")

        response_format = kwargs.get("response_format", self.response_format)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        data = {
            "model": kwargs.get("model", self.model),
            "response_format": response_format,
            "timestamps": kwargs.get("timestamps", self.timestamps),
        }

        if kwargs.get("language"):
            data["language"] = kwargs["language"]

        with open(audio_file, "rb") as f:
            files = {"file": f}
            response = requests.post(self.endpoint, headers=headers, data=data, files=files)

        if response.status_code == 200:
            if response_format == "json":
                return response.json()
            return response.text
        raise Exception(f"Transcription failed: {response.text}")

    def _validate_audio_file(self, audio_file: str):
        """
        Validates that the file exists and uses a Venice-supported audio format.
        """
        if not os.path.exists(audio_file):
            raise ValueError(f"File {audio_file} does not exist.")

        valid_extensions = [
            ".wav",
            ".wave",
            ".flac",
            ".m4a",
            ".aac",
            ".mp4",
            ".mp3",
            ".ogg",
            ".oga",
            ".webm",
        ]
        if not any(str(audio_file).lower().endswith(ext) for ext in valid_extensions):
            raise ValueError(
                f"Unsupported audio file format: {audio_file}. Supported formats are {valid_extensions}."
            )
