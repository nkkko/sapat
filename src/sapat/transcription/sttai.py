import mimetypes
import os

import requests
from dotenv import load_dotenv

from .base import TranscriptionBase


load_dotenv(".env")


class STTAITranscription(TranscriptionBase):
    """
    STT.ai API implementation for transcription.
    """

    def __init__(self, temperature: float, response_format: str = "json"):
        """
        Initializes the STTAITranscription class.

        Parameters:
        - temperature (float): Accepted for CLI compatibility. STT.ai does not use it.
        - response_format (str): Default response format for transcription.
        """
        self.api_key = os.getenv("STTAI_API_KEY")
        self.endpoint = os.getenv("STTAI_API_ENDPOINT", "https://api.stt.ai/v1/transcribe")
        self.model = os.getenv("STTAI_MODEL", "large-v3-turbo")
        self.diarize = os.getenv("STTAI_DIARIZE", "true")
        self.speakers = os.getenv("STTAI_SPEAKERS", "0")
        self.temperature = temperature
        self.response_format = os.getenv("STTAI_RESPONSE_FORMAT", response_format)
        self.max_file_size_mb = float(os.getenv("STTAI_MAX_FILE_SIZE_MB", "500"))

    def transcribe_audio(self, audio_file: str, **kwargs):
        """
        Transcribes an audio file using the STT.ai transcription API.

        Parameters:
        - audio_file (str): Path to the audio file.
        - kwargs: Additional parameters for transcription.

        Returns:
        - dict or str: The transcription result.
        """
        self._validate_audio_file(audio_file)

        response_format = kwargs.get("response_format", self.response_format)
        data = {
            "model": kwargs.get("model", self.model),
            "language": kwargs.get("language", "auto"),
            "diarize": kwargs.get("diarize", self.diarize),
            "speakers": str(kwargs.get("speakers", self.speakers)),
            "response_format": response_format,
        }
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        mime_type = mimetypes.guess_type(audio_file)[0] or "audio/mpeg"
        with open(audio_file, "rb") as f:
            files = {"file": (os.path.basename(audio_file), f, mime_type)}
            response = requests.post(self.endpoint, headers=headers, data=data, files=files)

        if response.status_code == 200:
            if response_format == "json":
                return response.json()
            return response.text

        raise Exception(f"STT.ai transcription failed: {response.text}")

    def _validate_audio_file(self, audio_file: str):
        """
        Validates the audio file for size and format.
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
            ".ogg",
            ".m4a",
            ".aac",
            ".opus",
            ".mp4",
            ".webm",
            ".mov",
        ]
        if not any(str(audio_file).lower().endswith(ext) for ext in valid_extensions):
            raise ValueError(f"Unsupported audio file format: {audio_file}. Supported formats are {valid_extensions}.")
