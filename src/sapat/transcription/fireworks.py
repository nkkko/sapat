import os
import requests
from dotenv import load_dotenv
from .base import TranscriptionBase


load_dotenv(".env")


class FireworksTranscription(TranscriptionBase):
    """
    Fireworks AI API implementation for pre-recorded audio transcription.
    """

    def __init__(self, temperature: float, response_format: str = "json"):
        self.api_key = os.getenv("FIREWORKS_API_KEY")
        self.model = os.getenv("FIREWORKS_MODEL", "whisper-v3")
        self.endpoint = os.getenv("FIREWORKS_API_ENDPOINT")
        self.vad_model = os.getenv("FIREWORKS_VAD_MODEL")
        self.alignment_model = os.getenv("FIREWORKS_ALIGNMENT_MODEL")
        self.preprocessing = os.getenv("FIREWORKS_PREPROCESSING")
        self.temperature = temperature
        self.response_format = response_format
        self.max_file_size_mb = 1024

    def transcribe_audio(self, audio_file: str, **kwargs):
        """
        Transcribes an audio file using the Fireworks AI Audio API.
        """
        self._validate_audio_file(audio_file)

        if not self.api_key:
            raise ValueError("FIREWORKS_API_KEY is required to use the Fireworks provider.")

        model = kwargs.get("model", self.model)
        data = {
            "model": model,
            "response_format": kwargs.get("response_format", self.response_format),
            "temperature": kwargs.get("temperature", self.temperature),
        }

        for key in ("language", "prompt", "timestamp_granularities", "diarize", "min_speakers", "max_speakers"):
            value = kwargs.get(key)
            if value not in (None, ""):
                data[key] = value

        self._add_optional_env_value(data, "vad_model", self.vad_model)
        self._add_optional_env_value(data, "alignment_model", self.alignment_model)
        self._add_optional_env_value(data, "preprocessing", self.preprocessing)

        headers = {"Authorization": self.api_key}
        with open(audio_file, "rb") as f:
            response = requests.post(
                self._endpoint_for_model(model),
                headers=headers,
                data=data,
                files={"file": f},
            )

        if response.status_code == 200:
            if data["response_format"] in ["json", "verbose_json"]:
                return response.json()
            return response.text
        raise Exception(f"Transcription failed: {response.text}")

    def generate_corrected_transcript(self, audio_file: str, temperature: float, system_prompt: str):
        """
        Fireworks ASR does not provide the chat correction path used by other providers.
        """
        raise NotImplementedError(
            "Correction is not implemented for the Fireworks provider. "
            "Run without --correct or use a provider with chat-completion support."
        )

    def _endpoint_for_model(self, model: str):
        if self.endpoint:
            return self.endpoint
        if model == "whisper-v3-turbo":
            return "https://audio-turbo.api.fireworks.ai/v1/audio/transcriptions"
        return "https://audio-prod.api.fireworks.ai/v1/audio/transcriptions"

    @staticmethod
    def _add_optional_env_value(data, key, value):
        if value not in (None, ""):
            data[key] = value

    def _validate_audio_file(self, audio_file: str):
        if not os.path.exists(audio_file):
            raise ValueError(f"File {audio_file} does not exist.")

        file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            raise Exception(f"File size exceeds the maximum limit of {self.max_file_size_mb} MB.")

        valid_extensions = [".mp3", ".wav", ".flac", ".m4a", ".ogg", ".webm", ".mp4"]
        if not any(str(audio_file).lower().endswith(ext) for ext in valid_extensions):
            raise ValueError(f"Unsupported audio file format: {audio_file}. Supported formats are {valid_extensions}.")
