import base64
import os

import requests
from dotenv import load_dotenv

from .base import TranscriptionBase


load_dotenv(".env")


class GoogleSpeechTranscription(TranscriptionBase):
    """
    Google Cloud Speech-to-Text REST API implementation.
    """

    def __init__(self, temperature: float, response_format: str = "json"):
        self.api_key = os.getenv("GOOGLE_SPEECH_API_KEY")
        self.endpoint = os.getenv(
            "GOOGLE_SPEECH_API_ENDPOINT",
            "https://speech.googleapis.com/v1p1beta1/speech:recognize",
        )
        self.model = os.getenv("GOOGLE_SPEECH_MODEL", "latest_long")
        self.default_language = os.getenv("GOOGLE_SPEECH_LANGUAGE", "en-US")
        self.use_enhanced = os.getenv("GOOGLE_SPEECH_USE_ENHANCED", "false").lower() == "true"
        self.temperature = temperature
        self.response_format = response_format

    def transcribe_audio(self, audio_file: str, **kwargs):
        if not self.api_key:
            raise ValueError("GOOGLE_SPEECH_API_KEY is required for Google transcription.")

        language = kwargs.get("language") or self.default_language
        response_format = kwargs.get("response_format", self.response_format)

        with open(audio_file, "rb") as f:
            audio_content = base64.b64encode(f.read()).decode("utf-8")

        request_body = {
            "config": {
                "encoding": "MP3",
                "languageCode": self._normalize_language(language),
                "enableAutomaticPunctuation": True,
                "model": kwargs.get("model", self.model),
                "useEnhanced": kwargs.get("use_enhanced", self.use_enhanced),
            },
            "audio": {
                "content": audio_content,
            },
        }

        response = requests.post(
            self.endpoint,
            params={"key": self.api_key},
            json=request_body,
            timeout=120,
        )

        if response.status_code != 200:
            raise Exception(f"Transcription failed: {response.text}")

        payload = response.json()
        text = self._extract_transcript(payload)

        if response_format in ["json", "verbose_json"]:
            return {
                "text": text,
                "results": payload.get("results", []),
            }
        return text

    @staticmethod
    def _normalize_language(language: str):
        language_map = {
            "en": "en-US",
            "es": "es-ES",
            "fr": "fr-FR",
            "de": "de-DE",
            "pt": "pt-BR",
        }
        return language_map.get(language, language)

    @staticmethod
    def _extract_transcript(payload):
        transcripts = []
        for result in payload.get("results", []):
            alternatives = result.get("alternatives", [])
            if alternatives:
                transcript = alternatives[0].get("transcript", "").strip()
                if transcript:
                    transcripts.append(transcript)
        return "\n".join(transcripts)

    @staticmethod
    def generate_corrected_transcript(audio_file, temperature, prompt):
        raise NotImplementedError("Correction not implemented for Google Speech-to-Text.")
