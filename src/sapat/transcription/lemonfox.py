import os

import requests
from dotenv import load_dotenv

from .base import TranscriptionBase

load_dotenv(".env")


class LemonfoxTranscription(TranscriptionBase):
    """
    Lemonfox Speech-to-Text implementation.
    """

    def __init__(self, temperature: float, response_format: str = "json"):
        """
        Initializes the LemonfoxTranscription class.

        Parameters:
        - temperature (float): Accepted for CLI compatibility with other providers.
        - response_format (str): Default response format for transcription.
        """
        self.api_key = os.getenv("LEMONFOX_API_KEY")
        self.endpoint = os.getenv(
            "LEMONFOX_API_ENDPOINT",
            "https://api.lemonfox.ai/v1/audio/transcriptions",
        )
        self.response_format = os.getenv("LEMONFOX_RESPONSE_FORMAT", response_format)
        self.temperature = temperature
        self.max_file_size_mb = 100

    def transcribe_audio(self, audio_file: str, **kwargs):
        """
        Transcribes an audio file using Lemonfox Speech-to-Text.

        Parameters:
        - audio_file (str): Path to the audio file.
        - kwargs: Additional parameters for transcription.

        Returns:
        - dict or str: The transcription result.
        """
        self._validate_audio_file(audio_file)

        if not self.api_key:
            raise ValueError("LEMONFOX_API_KEY is required for Lemonfox transcription.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        data = {
            "response_format": kwargs.get("response_format", self.response_format),
        }

        if kwargs.get("language"):
            data["language"] = kwargs["language"]
        if kwargs.get("prompt"):
            data["prompt"] = kwargs["prompt"]

        with open(audio_file, "rb") as f:
            files = {"file": f}
            response = requests.post(
                self.endpoint,
                headers=headers,
                data=data,
                files=files,
            )

        if response.status_code == 200:
            if data["response_format"] in ["json", "verbose_json"]:
                return response.json()
            return response.text
        raise Exception(f"Transcription failed: {response.text}")

    def generate_corrected_transcript(self, audio_file: str, temperature: float, system_prompt: str):
        """
        Lemonfox Speech-to-Text does not provide a transcript-correction chat API.
        """
        raise NotImplementedError("Correction is not implemented for Lemonfox.")

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
            ".aac",
            ".opus",
            ".ogg",
            ".m4a",
            ".mp4",
            ".mpeg",
            ".mov",
            ".webm",
        ]
        if not any(str(audio_file).endswith(ext) for ext in valid_extensions):
            raise ValueError(
                f"Unsupported audio file format: {audio_file}. Supported formats are {valid_extensions}."
            )
