import os

import requests
from dotenv import load_dotenv

from .base import TranscriptionBase


load_dotenv(".env")


class IBMWatsonTranscription(TranscriptionBase):
    """
    IBM Watson Speech to Text implementation for transcription.
    """

    def __init__(self, temperature: float):
        self.api_key = os.getenv("IBM_WATSON_STT_API_KEY")
        self.service_url = os.getenv("IBM_WATSON_STT_URL")
        self.model = os.getenv("IBM_WATSON_STT_MODEL", "en-US_BroadbandModel")
        self.smart_formatting = os.getenv("IBM_WATSON_STT_SMART_FORMATTING", "true")
        self.temperature = temperature

    def transcribe_audio(self, audio_file: str, **kwargs):
        """
        Transcribes an audio file using IBM Watson Speech to Text.

        Parameters:
        - audio_file (str): Path to the audio file.
        - kwargs: Additional parameters for transcription.

        Returns:
        - str: The transcript text.
        """
        self._validate_audio_file(audio_file)
        self._validate_config()

        url = f"{self.service_url.rstrip('/')}/v1/recognize"
        params = {
            "model": kwargs.get("model", self.model),
            "smart_formatting": self.smart_formatting,
        }
        headers = {
            "Content-Type": self._content_type(audio_file),
        }

        with open(audio_file, "rb") as f:
            response = requests.post(
                url,
                auth=("apikey", self.api_key),
                headers=headers,
                params=params,
                data=f,
            )

        if not response.ok:
            raise Exception(f"Transcription failed: {response.status_code} {response.text}")

        return self._extract_transcript(response.json())

    def _extract_transcript(self, result):
        transcripts = []
        for item in result.get("results", []):
            alternatives = item.get("alternatives") or []
            if alternatives:
                transcript = alternatives[0].get("transcript")
                if transcript:
                    transcripts.append(transcript.strip())

        if transcripts:
            return " ".join(transcripts).strip()

        raise Exception("IBM Watson response did not include a transcript.")

    def _validate_config(self):
        if not self.api_key:
            raise ValueError("IBM_WATSON_STT_API_KEY is required for IBM Watson transcription.")
        if not self.service_url:
            raise ValueError("IBM_WATSON_STT_URL is required for IBM Watson transcription.")

    def _validate_audio_file(self, audio_file: str):
        if not os.path.exists(audio_file):
            raise ValueError(f"File {audio_file} does not exist.")

        if self._content_type(audio_file) is None:
            valid_extensions = [".mp3", ".wav", ".flac", ".ogg", ".m4a"]
            raise ValueError(
                f"Unsupported audio file format: {audio_file}. Supported formats are {valid_extensions}."
            )

    def _content_type(self, audio_file: str):
        extension = os.path.splitext(audio_file)[1].lower()
        return {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
            ".m4a": "audio/mp4",
        }.get(extension)
