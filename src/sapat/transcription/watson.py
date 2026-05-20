import mimetypes
import os

import requests
from dotenv import load_dotenv

from .base import TranscriptionBase


load_dotenv(".env")


class IBMWatsonTranscription(TranscriptionBase):
    """
    IBM Watson Speech to Text implementation.
    """

    def __init__(self, temperature: float):
        self.api_key = os.getenv("IBM_WATSON_STT_API_KEY")
        self.endpoint = os.getenv("IBM_WATSON_STT_URL")
        self.model = os.getenv("IBM_WATSON_STT_MODEL")
        self.temperature = temperature
        self.max_file_size_mb = 100

    def transcribe_audio(self, audio_file: str, **kwargs):
        """
        Transcribes an audio file using IBM Watson Speech to Text.
        """
        self._validate_config()
        self._validate_audio_file(audio_file)

        params = {}
        if self.model:
            params["model"] = self.model

        content_type = mimetypes.guess_type(audio_file)[0] or "audio/mpeg"
        recognize_url = self._recognize_url()

        with open(audio_file, "rb") as audio:
            response = requests.post(
                recognize_url,
                auth=("apikey", self.api_key),
                headers={"Content-Type": content_type},
                params=params,
                data=audio,
            )

        if response.status_code != 200:
            raise Exception(f"Transcription failed: {response.text}")

        return self._extract_transcript(response.json())

    def _recognize_url(self):
        endpoint = self.endpoint.rstrip("/")
        if endpoint.endswith("/v1/recognize"):
            return endpoint
        return endpoint + "/v1/recognize"

    def _validate_config(self):
        if not self.api_key:
            raise ValueError("IBM_WATSON_STT_API_KEY is required for IBM Watson transcription.")
        if not self.endpoint:
            raise ValueError("IBM_WATSON_STT_URL is required for IBM Watson transcription.")

    def _validate_audio_file(self, audio_file: str):
        if not os.path.exists(audio_file):
            raise ValueError(f"File {audio_file} does not exist.")

        file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            raise Exception(f"File size exceeds the maximum limit of {self.max_file_size_mb} MB.")

        valid_extensions = [".mp3", ".wav", ".flac", ".ogg", ".webm"]
        if not any(str(audio_file).endswith(ext) for ext in valid_extensions):
            raise ValueError(f"Unsupported audio file format: {audio_file}. Supported formats are {valid_extensions}.")

    @staticmethod
    def _extract_transcript(payload):
        transcripts = []
        for result in payload.get("results", []):
            alternatives = result.get("alternatives") or []
            if alternatives:
                transcripts.append(alternatives[0].get("transcript", "").strip())
        return " ".join(part for part in transcripts if part)
