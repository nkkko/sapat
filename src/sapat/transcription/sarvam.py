import os

import requests
from dotenv import load_dotenv

from .base import TranscriptionBase


load_dotenv(".env")


class SarvamTranscription(TranscriptionBase):
    """
    Sarvam AI implementation for speech-to-text transcription.
    """

    LANGUAGE_ALIASES = {
        "as": "as-IN",
        "bn": "bn-IN",
        "brx": "brx-IN",
        "doi": "doi-IN",
        "en": "en-IN",
        "gu": "gu-IN",
        "hi": "hi-IN",
        "kn": "kn-IN",
        "kok": "kok-IN",
        "ks": "ks-IN",
        "mai": "mai-IN",
        "ml": "ml-IN",
        "mni": "mni-IN",
        "mr": "mr-IN",
        "ne": "ne-IN",
        "od": "od-IN",
        "pa": "pa-IN",
        "sa": "sa-IN",
        "sat": "sat-IN",
        "sd": "sd-IN",
        "ta": "ta-IN",
        "te": "te-IN",
        "ur": "ur-IN",
    }

    def __init__(self, temperature: float):
        """
        Initializes the SarvamTranscription class.

        Parameters:
        - temperature (float): Accepted for CLI compatibility with other providers.
        """
        self.api_key = os.getenv("SARVAM_API_KEY")
        self.endpoint = os.getenv("SARVAM_STT_ENDPOINT") or "https://api.sarvam.ai/speech-to-text"
        self.model = os.getenv("SARVAM_STT_MODEL") or "saaras:v3"
        self.mode = os.getenv("SARVAM_STT_MODE") or "transcribe"
        self.default_language_code = os.getenv("SARVAM_LANGUAGE_CODE") or None
        self.temperature = temperature

    def transcribe_audio(self, audio_file: str, **kwargs):
        """
        Transcribes an audio file using Sarvam AI's speech-to-text REST API.

        Parameters:
        - audio_file (str): Path to the audio file.
        - kwargs: Additional transcription parameters.

        Returns:
        - dict: The Sarvam response with a text alias for Sapat's writer.
        """
        self._validate_audio_file(audio_file)

        if not self.api_key:
            raise ValueError("SARVAM_API_KEY is required for Sarvam transcription.")

        data = {
            "model": kwargs.get("model", self.model),
            "mode": kwargs.get("mode", self.mode),
        }

        language_code = self._language_code(kwargs.get("language"))
        if language_code:
            data["language_code"] = language_code

        headers = {
            "api-subscription-key": self.api_key,
        }

        with open(audio_file, "rb") as f:
            files = {"file": f}
            response = requests.post(self.endpoint, headers=headers, data=data, files=files)

        if response.status_code != 200:
            raise Exception(f"Transcription failed: {response.text}")

        payload = response.json()
        transcript = payload.get("transcript", "")
        payload["text"] = transcript
        return payload

    def _language_code(self, language):
        language = language or self.default_language_code
        if not language:
            return None

        normalized = language.strip()
        if not normalized or normalized.lower() == "auto":
            return None
        if normalized.lower() == "unknown":
            return "unknown"
        if "-" in normalized:
            return normalized
        return self.LANGUAGE_ALIASES.get(normalized.lower(), normalized)

    @staticmethod
    def _validate_audio_file(audio_file: str):
        """
        Validates that the audio file exists and uses a Sarvam-supported extension.
        """
        if not os.path.exists(audio_file):
            raise ValueError(f"File {audio_file} does not exist.")

        valid_extensions = [
            ".aac",
            ".aiff",
            ".amr",
            ".flac",
            ".m4a",
            ".mp3",
            ".mp4",
            ".ogg",
            ".opus",
            ".pcm",
            ".wav",
            ".webm",
            ".wma",
        ]
        if not any(str(audio_file).lower().endswith(ext) for ext in valid_extensions):
            raise ValueError(
                f"Unsupported audio file format: {audio_file}. Supported formats are {valid_extensions}."
            )
