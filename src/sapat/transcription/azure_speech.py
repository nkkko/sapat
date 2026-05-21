import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

from .base import TranscriptionBase

# Load environment variables
load_dotenv(".env")


class AzureSpeechTranscription(TranscriptionBase):
    """
    Azure AI Speech implementation for fast transcription.
    """

    DEFAULT_API_VERSION = "2025-10-15"
    DEFAULT_MAX_FILE_SIZE_MB = 250
    LANGUAGE_LOCALES = {
        "en": "en-US",
        "es": "es-ES",
        "fr": "fr-FR",
        "de": "de-DE",
        "it": "it-IT",
        "pt": "pt-BR",
        "ja": "ja-JP",
        "ko": "ko-KR",
        "zh": "zh-CN",
    }

    def __init__(self, temperature: float, response_format: str = "text"):
        """
        Initializes the AzureSpeechTranscription class.

        Parameters:
        - temperature (float): Kept for CLI parity with the other providers.
        - response_format (str): Kept for CLI parity. Azure Speech returns JSON.
        """
        self.api_key = os.getenv("AZURE_SPEECH_API_KEY") or os.getenv("AZURE_SPEECH_KEY")
        self.region = os.getenv("AZURE_SPEECH_REGION")
        self.endpoint = self._normalize_endpoint(os.getenv("AZURE_SPEECH_ENDPOINT"))
        self.api_version = os.getenv("AZURE_SPEECH_API_VERSION", self.DEFAULT_API_VERSION)
        self.temperature = temperature
        self.response_format = response_format
        self.max_file_size_mb = int(
            os.getenv("AZURE_SPEECH_MAX_FILE_SIZE_MB", str(self.DEFAULT_MAX_FILE_SIZE_MB))
        )

    def transcribe_audio(self, audio_file: str, **kwargs):
        """
        Transcribes an audio file using Azure AI Speech fast transcription.

        Parameters:
        - audio_file (str): Path to the audio file.
        - kwargs: Additional parameters for transcription.

        Returns:
        - str: The combined transcription text.
        """
        self._validate_configuration()
        self._validate_audio_file(audio_file)

        definition = self._build_definition(language=kwargs.get("language"))

        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
        }
        data = {
            "definition": json.dumps(definition),
        }

        with open(audio_file, "rb") as f:
            files = {
                "audio": (Path(audio_file).name, f, "application/octet-stream"),
            }
            response = requests.post(self._transcribe_url(), headers=headers, data=data, files=files)

        if response.status_code == 200:
            return self._extract_text(response.json())

        raise Exception(f"Transcription failed: {response.text}")

    def _transcribe_url(self):
        endpoint = self.endpoint or f"https://{self.region}.api.cognitive.microsoft.com"
        return (
            f"{endpoint}/speechtotext/transcriptions:transcribe"
            f"?api-version={self.api_version}"
        )

    def _build_definition(self, language=None):
        locale = self._language_to_locale(language)
        return {"locales": [locale]}

    def _language_to_locale(self, language=None):
        if not language:
            return "en-US"

        normalized = language.strip()
        if "-" in normalized:
            return normalized

        return self.LANGUAGE_LOCALES.get(normalized.lower(), normalized)

    def _extract_text(self, payload):
        combined = payload.get("combinedPhrases") or []
        combined_text = [
            phrase.get("text", "").strip()
            for phrase in combined
            if phrase.get("text", "").strip()
        ]
        if combined_text:
            return "\n".join(combined_text)

        phrases = payload.get("phrases") or []
        phrase_text = [
            phrase.get("text", "").strip()
            for phrase in phrases
            if phrase.get("text", "").strip()
        ]
        return "\n".join(phrase_text)

    def _validate_configuration(self):
        missing = []
        if not self.api_key:
            missing.append("AZURE_SPEECH_API_KEY or AZURE_SPEECH_KEY")
        if not self.endpoint and not self.region:
            missing.append("AZURE_SPEECH_ENDPOINT or AZURE_SPEECH_REGION")

        if missing:
            raise ValueError(
                "Missing Azure Speech configuration: " + ", ".join(missing)
            )

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
            ".ogg",
            ".opus",
            ".wma",
            ".aac",
            ".amr",
            ".webm",
        ]
        if not any(str(audio_file).lower().endswith(ext) for ext in valid_extensions):
            raise ValueError(
                f"Unsupported audio file format: {audio_file}. "
                f"Supported formats are {valid_extensions}."
            )

    @staticmethod
    def _normalize_endpoint(endpoint):
        if not endpoint:
            return None
        return endpoint.rstrip("/")
