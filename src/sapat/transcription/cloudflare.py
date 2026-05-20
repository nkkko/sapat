import os
from pathlib import Path

import requests
from dotenv import load_dotenv

from .base import TranscriptionBase

# Load environment variables
load_dotenv(".env")


class CloudflareTranscription(TranscriptionBase):
    """
    Cloudflare Workers AI implementation for transcription.
    """

    def __init__(self, temperature: float, response_format: str = "json"):
        """
        Initializes the CloudflareTranscription class.

        Parameters:
        - temperature (float): Default temperature value for transcription.
        - response_format (str): Default response format for transcription.
        """
        self.api_token = os.getenv("CLOUDFLARE_API_TOKEN")
        self.account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
        self.model = os.getenv("CLOUDFLARE_WHISPER_MODEL", "@cf/openai/whisper")
        self.endpoint = os.getenv("CLOUDFLARE_API_ENDPOINT")
        self.temperature = temperature
        self.response_format = response_format
        self.max_file_size_mb = int(os.getenv("CLOUDFLARE_MAX_FILE_SIZE_MB", "25"))

    def transcribe_audio(self, audio_file: str, **kwargs):
        """
        Transcribes an audio file using Cloudflare Workers AI Whisper.

        Parameters:
        - audio_file (str): Path to the audio file.
        - kwargs: Additional parameters accepted for CLI compatibility.

        Returns:
        - dict or str: The transcription result.
        """
        self._validate_audio_file(audio_file)
        self._validate_configuration()

        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": self._content_type(audio_file),
        }

        with open(audio_file, "rb") as f:
            response = requests.post(self._transcription_endpoint(), headers=headers, data=f.read())

        if response.status_code != 200:
            raise Exception(f"Transcription failed: {response.text}")

        payload = response.json()
        if isinstance(payload, dict) and "result" in payload:
            return payload["result"]
        return payload

    def _transcription_endpoint(self):
        if self.endpoint:
            return self.endpoint

        return f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{self.model}"

    def _validate_configuration(self):
        if not self.api_token:
            raise ValueError("CLOUDFLARE_API_TOKEN is required for Cloudflare transcription.")

        if not self.account_id and not self.endpoint:
            raise ValueError("CLOUDFLARE_ACCOUNT_ID is required unless CLOUDFLARE_API_ENDPOINT is set.")

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

        valid_extensions = [".mp3", ".wav", ".flac"]
        if not any(str(audio_file).endswith(ext) for ext in valid_extensions):
            raise ValueError(f"Unsupported audio file format: {audio_file}. Supported formats are {valid_extensions}.")

    @staticmethod
    def _content_type(audio_file: str):
        extension = Path(audio_file).suffix.lower()
        content_types = {
            ".flac": "audio/flac",
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
        }
        return content_types.get(extension, "application/octet-stream")
