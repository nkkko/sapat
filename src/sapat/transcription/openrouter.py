import base64
import os

import requests
from dotenv import load_dotenv

from .base import TranscriptionBase


load_dotenv(".env")


class OpenRouterTranscription(TranscriptionBase):
    """
    OpenRouter API implementation for speech-to-text transcription.
    """

    def __init__(self, temperature: float):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.endpoint = os.getenv(
            "OPENROUTER_API_ENDPOINT",
            "https://openrouter.ai/api/v1/audio/transcriptions",
        )
        self.model = os.getenv("OPENROUTER_MODEL", "openai/whisper-large-v3")
        self.http_referer = os.getenv("OPENROUTER_HTTP_REFERER")
        self.app_title = os.getenv("OPENROUTER_APP_TITLE")
        self.temperature = temperature
        self.max_file_size_mb = 25

    def transcribe_audio(self, audio_file: str, **kwargs):
        """
        Transcribes an audio file using OpenRouter's transcription API.

        Parameters:
        - audio_file (str): Path to the audio file.
        - kwargs: Additional parameters for transcription.

        Returns:
        - dict: The transcription response.
        """
        self._validate_audio_file(audio_file)

        with open(audio_file, "rb") as f:
            encoded_audio = base64.b64encode(f.read()).decode("ascii")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.app_title:
            headers["X-Title"] = self.app_title

        payload = {
            "input_audio": {
                "data": encoded_audio,
                "format": os.path.splitext(audio_file)[1].lstrip(".").lower(),
            },
            "model": kwargs.get("model", self.model),
            "temperature": kwargs.get("temperature", self.temperature),
        }

        language = kwargs.get("language")
        if language:
            payload["language"] = language

        response = requests.post(self.endpoint, headers=headers, json=payload)

        if response.status_code == 200:
            return response.json()

        raise Exception(f"Transcription failed: {response.text}")

    def generate_corrected_transcript(self, audio_file: str, temperature: float, system_prompt: str):
        raise NotImplementedError("Correction is not implemented for OpenRouter.")

    def _validate_audio_file(self, audio_file: str):
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required.")

        if not os.path.exists(audio_file):
            raise ValueError(f"File {audio_file} does not exist.")

        file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            raise Exception(f"File size exceeds the maximum limit of {self.max_file_size_mb} MB.")

        valid_extensions = [".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg"]
        if not any(str(audio_file).lower().endswith(ext) for ext in valid_extensions):
            raise ValueError(
                f"Unsupported audio file format: {audio_file}. Supported formats are {valid_extensions}."
            )
