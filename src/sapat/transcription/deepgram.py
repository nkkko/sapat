import os
from pathlib import Path

import requests
from dotenv import load_dotenv

from .base import TranscriptionBase

load_dotenv(".env")


class DeepgramTranscription(TranscriptionBase):
    """
    Deepgram API implementation for transcription.
    """

    def __init__(self, temperature: float):
        self.api_key = os.getenv("DEEPGRAM_API_KEY")
        self.model = os.getenv("DEEPGRAM_MODEL", "nova-3")
        self.endpoint = os.getenv(
            "DEEPGRAM_API_ENDPOINT",
            "https://api.deepgram.com/v1/listen",
        )
        self.temperature = temperature

    def transcribe_audio(self, audio_file: str, **kwargs):
        """
        Transcribes an audio file with Deepgram's prerecorded audio endpoint.
        """
        self._validate_audio_file(audio_file)

        params = {
            "model": kwargs.get("model", self.model),
            "smart_format": "true",
        }

        language = kwargs.get("language")
        if language:
            params["language"] = language

        prompt = kwargs.get("prompt")
        if prompt:
            params["keywords"] = prompt

        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": self._content_type(audio_file),
        }

        with open(audio_file, "rb") as f:
            response = requests.post(
                self.endpoint,
                headers=headers,
                params=params,
                data=f,
                timeout=120,
            )

        if response.status_code != 200:
            raise Exception(f"Transcription failed: {response.text}")

        payload = response.json()
        return self._extract_transcript(payload)

    @staticmethod
    def _content_type(audio_file: str):
        suffix = Path(audio_file).suffix.lower()
        if suffix == ".mp3":
            return "audio/mpeg"
        if suffix == ".wav":
            return "audio/wav"
        if suffix == ".flac":
            return "audio/flac"
        return "application/octet-stream"

    @staticmethod
    def _extract_transcript(payload):
        channels = payload.get("results", {}).get("channels", [])
        if not channels:
            return ""

        alternatives = channels[0].get("alternatives", [])
        if not alternatives:
            return ""

        return alternatives[0].get("transcript", "")

    @staticmethod
    def _validate_audio_file(audio_file: str):
        if not os.path.exists(audio_file):
            raise ValueError(f"File {audio_file} does not exist.")

        valid_extensions = [".mp3", ".wav", ".flac"]
        if not any(str(audio_file).endswith(ext) for ext in valid_extensions):
            raise ValueError(
                f"Unsupported audio file format: {audio_file}. "
                f"Supported formats are {valid_extensions}."
            )
