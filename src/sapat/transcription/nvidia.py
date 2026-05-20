import os

import requests
from dotenv import load_dotenv

from .base import TranscriptionBase

# Load environment variables
load_dotenv(".env")


class NvidiaTranscription(TranscriptionBase):
    """
    NVIDIA ASR NIM implementation for transcription.
    """

    def __init__(self, temperature: float, response_format: str = "json"):
        """
        Initializes the NvidiaTranscription class.

        Parameters:
        - temperature (float): Default temperature value for transcription.
        - response_format (str): Default response format for transcription.
        """
        self.api_key = os.getenv("NVIDIA_NIM_API_KEY")
        self.endpoint = os.getenv(
            "NVIDIA_NIM_BASE_URL",
            "https://integrate.api.nvidia.com/v1",
        ).rstrip("/")
        self.model = os.getenv("NVIDIA_NIM_MODEL", "nvidia/parakeet-ctc-0.6b-asr")
        self.temperature = temperature
        self.response_format = response_format
        self.max_file_size_mb = 25

    def transcribe_audio(self, audio_file: str, **kwargs):
        """
        Transcribes an audio file using the NVIDIA ASR NIM OpenAI-compatible API.

        Parameters:
        - audio_file (str): Path to the audio file.
        - kwargs: Additional parameters for transcription.

        Returns:
        - dict or str: The transcription result.
        """
        self._validate_configuration()
        self._validate_audio_file(audio_file)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        data = {
            "model": kwargs.get("model", self.model),
            "response_format": kwargs.get("response_format", self.response_format),
        }

        if "language" in kwargs:
            data["language"] = kwargs["language"]
        if "prompt" in kwargs:
            data["prompt"] = kwargs["prompt"]

        with open(audio_file, "rb") as f:
            files = {"file": f}
            response = requests.post(
                f"{self.endpoint}/audio/transcriptions",
                headers=headers,
                data=data,
                files=files,
            )

        if response.status_code == 200:
            if data["response_format"] in ["json", "verbose_json"]:
                return response.json()
            return response.text
        raise Exception(f"Transcription failed: {response.text}")

    def _validate_configuration(self):
        if not self.api_key:
            raise ValueError("NVIDIA_NIM_API_KEY is required for NVIDIA transcription.")

    def _validate_audio_file(self, audio_file: str):
        if not os.path.exists(audio_file):
            raise ValueError(f"File {audio_file} does not exist.")

        file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            raise Exception(
                f"File size exceeds the maximum limit of {self.max_file_size_mb} MB."
            )

        valid_extensions = [".mp3", ".wav", ".flac"]
        if not any(str(audio_file).endswith(ext) for ext in valid_extensions):
            raise ValueError(
                f"Unsupported audio file format: {audio_file}. "
                f"Supported formats are {valid_extensions}."
            )
