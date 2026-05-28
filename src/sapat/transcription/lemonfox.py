import os

import requests
from dotenv import load_dotenv

from .base import TranscriptionBase


load_dotenv(".env")


class LemonfoxTranscription(TranscriptionBase):
    """
    Lemonfox API implementation for transcription.
    """

    LANGUAGE_ALIASES = {
        "en": "english",
        "zh": "chinese",
        "de": "german",
        "es": "spanish",
        "ru": "russian",
        "ko": "korean",
        "fr": "french",
        "ja": "japanese",
        "pt": "portuguese",
        "tr": "turkish",
        "pl": "polish",
        "nl": "dutch",
        "ar": "arabic",
        "sv": "swedish",
        "it": "italian",
        "id": "indonesian",
        "hi": "hindi",
        "fi": "finnish",
        "vi": "vietnamese",
        "he": "hebrew",
        "uk": "ukrainian",
        "el": "greek",
    }

    def __init__(self, temperature: float, response_format: str = "json"):
        """
        Initializes the LemonfoxTranscription class.

        Parameters:
        - temperature (float): Accepted for CLI compatibility. Lemonfox does not
          currently expose a temperature parameter for transcription.
        - response_format (str): Default response format for transcription.
        """
        self.api_key = os.getenv("LEMONFOX_API_KEY")
        self.endpoint = os.getenv(
            "LEMONFOX_API_ENDPOINT",
            "https://api.lemonfox.ai/v1/audio/transcriptions",
        )
        self.temperature = temperature
        self.response_format = os.getenv("LEMONFOX_RESPONSE_FORMAT", response_format)
        self.max_file_size_mb = 100

    def transcribe_audio(self, audio_file: str, **kwargs):
        """
        Transcribes an audio file using Lemonfox's OpenAI-compatible API.

        Parameters:
        - audio_file (str): Path to the audio file.
        - kwargs: Additional parameters for transcription.

        Returns:
        - dict or str: The transcription result.
        """
        self._validate_api_key()
        self._validate_audio_file(audio_file)

        response_format = kwargs.get("response_format", self.response_format)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {"response_format": response_format}

        language = self._normalize_language(kwargs.get("language"))
        if language:
            data["language"] = language

        prompt = kwargs.get("prompt")
        if prompt:
            data["prompt"] = prompt

        if os.getenv("LEMONFOX_SPEAKER_LABELS", "").lower() in {"1", "true", "yes"}:
            data["speaker_labels"] = "true"
            if response_format != "verbose_json":
                data["response_format"] = "verbose_json"
                response_format = "verbose_json"

        if os.getenv("LEMONFOX_TRANSLATE", "").lower() in {"1", "true", "yes"}:
            data["translate"] = "true"

        with open(audio_file, "rb") as f:
            response = requests.post(
                self.endpoint,
                headers=headers,
                data=data,
                files={"file": f},
                timeout=120,
            )

        if response.status_code != 200:
            raise Exception(
                f"Lemonfox transcription failed ({response.status_code}): {response.text}"
            )

        if response_format in ["json", "verbose_json"]:
            return response.json()
        return response.text

    def _validate_api_key(self):
        if not self.api_key:
            raise ValueError("LEMONFOX_API_KEY is required to use the Lemonfox API.")

    def _validate_audio_file(self, audio_file: str):
        if not os.path.exists(audio_file):
            raise ValueError(f"File {audio_file} does not exist.")

        file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            raise Exception(
                f"File size exceeds the maximum limit of {self.max_file_size_mb} MB."
            )

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
        if not any(str(audio_file).lower().endswith(ext) for ext in valid_extensions):
            raise ValueError(
                f"Unsupported audio file format: {audio_file}. "
                f"Supported formats are {valid_extensions}."
            )

    def _normalize_language(self, language):
        if not language:
            return None
        return self.LANGUAGE_ALIASES.get(str(language).lower(), language)
