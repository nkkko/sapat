import os

import requests
from dotenv import load_dotenv

from .base import TranscriptionBase

load_dotenv(".env")


class HuggingFaceTranscription(TranscriptionBase):
    """
    Hugging Face Inference API implementation for automatic speech recognition.
    """

    def __init__(self, temperature: float):
        self.api_key = os.getenv("HUGGINGFACE_API_KEY")
        self.model = os.getenv("HUGGINGFACE_MODEL", "openai/whisper-large-v3-turbo")
        self.endpoint = os.getenv("HUGGINGFACE_API_ENDPOINT")
        self.temperature = temperature
        self.max_file_size_mb = 25

    def transcribe_audio(self, audio_file: str, **kwargs):
        """
        Transcribes an audio file using the Hugging Face Inference API.

        Parameters:
        - audio_file (str): Path to the audio file.

        Returns:
        - dict or str: The transcription result.
        """
        self._validate_audio_file(audio_file)

        if not self.api_key:
            raise ValueError("HUGGINGFACE_API_KEY must be set.")

        model = kwargs.get("model", self.model)
        endpoint = self.endpoint or f"https://api-inference.huggingface.co/models/{model}"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": self._content_type_for_audio(audio_file),
            "X-Wait-For-Model": "true",
        }

        with open(audio_file, "rb") as f:
            response = requests.post(endpoint, headers=headers, data=f.read())

        if response.status_code != 200:
            raise Exception(f"Transcription failed: {response.text}")

        try:
            result = response.json()
        except ValueError:
            return response.text

        if isinstance(result, dict) and "text" in result:
            return result

        if isinstance(result, list):
            text = " ".join(item.get("text", "") for item in result if isinstance(item, dict))
            if text:
                return {"text": text}

        return result

    def _validate_audio_file(self, audio_file: str):
        if not os.path.exists(audio_file):
            raise ValueError(f"File {audio_file} does not exist.")

        file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            raise Exception(f"File size exceeds the maximum limit of {self.max_file_size_mb} MB.")

        valid_extensions = [".mp3", ".wav", ".flac"]
        if not any(str(audio_file).lower().endswith(ext) for ext in valid_extensions):
            raise ValueError(f"Unsupported audio file format: {audio_file}. Supported formats are {valid_extensions}.")

    def _content_type_for_audio(self, audio_file: str):
        extension = os.path.splitext(audio_file)[1].lower()
        return {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".flac": "audio/flac",
        }[extension]
