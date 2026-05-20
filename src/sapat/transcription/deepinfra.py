import os

import requests
from dotenv import load_dotenv
from openai import OpenAI

from .base import TranscriptionBase


load_dotenv(".env")


class DeepInfraTranscription(TranscriptionBase):
    """
    DeepInfra API implementation for OpenAI-compatible transcription.
    """

    def __init__(self, temperature: float, response_format: str = "json"):
        self.api_key = os.getenv("DEEPINFRA_TOKEN")
        self.model = os.getenv("DEEPINFRA_MODEL", "openai/whisper-large")
        self.model_name_chat = os.getenv(
            "DEEPINFRA_MODEL_NAME_CHAT",
            "deepseek-ai/DeepSeek-V3",
        )
        self.endpoint = os.getenv(
            "DEEPINFRA_API_ENDPOINT",
            "https://api.deepinfra.com/v1/audio/transcriptions",
        )
        self.openai_base_url = os.getenv(
            "DEEPINFRA_OPENAI_BASE_URL",
            "https://api.deepinfra.com/v1/openai",
        )
        self.temperature = temperature
        self.response_format = response_format
        self.max_file_size_mb = 25

    def transcribe_audio(self, audio_file: str, **kwargs):
        self._validate_audio_file(audio_file)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        data = {
            "model": kwargs.get("model", self.model),
            "response_format": kwargs.get("response_format", self.response_format),
            "temperature": kwargs.get("temperature", self.temperature),
        }

        if "language" in kwargs:
            data["language"] = kwargs["language"]
        if "prompt" in kwargs:
            data["prompt"] = kwargs["prompt"]

        with open(audio_file, "rb") as f:
            files = {"file": f}
            response = requests.post(
                self.endpoint, headers=headers, data=data, files=files
            )

        if response.status_code == 200:
            if data["response_format"] in ["json", "verbose_json"]:
                return response.json()
            return response.text
        raise Exception(f"Transcription failed: {response.text}")

    def generate_corrected_transcript(
        self, audio_file: str, temperature: float, system_prompt: str
    ):
        client = OpenAI(api_key=self.api_key, base_url=self.openai_base_url)

        transcription = self.transcribe_audio(audio_file)
        transcription_text = (
            transcription.get("text", "") if isinstance(transcription, dict)
            else transcription
        )

        response = client.chat.completions.create(
            model=self.model_name_chat,
            temperature=temperature,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": transcription_text,
                },
            ],
        )
        return response.choices[0].message.content

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
