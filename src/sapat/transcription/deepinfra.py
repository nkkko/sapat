import os

import requests
from dotenv import load_dotenv
from openai import OpenAI

from .base import TranscriptionBase

load_dotenv(".env")


class DeepInfraTranscription(TranscriptionBase):
    """
    DeepInfra native speech-recognition implementation.
    """

    def __init__(self, temperature: float, response_format: str = "json"):
        self.api_key = os.getenv("DEEPINFRA_TOKEN")
        self.model = os.getenv("DEEPINFRA_MODEL", "openai/whisper-large")
        self.endpoint = os.getenv("DEEPINFRA_API_ENDPOINT")
        self.model_name_chat = os.getenv("DEEPINFRA_MODEL_NAME_CHAT")
        self.openai_endpoint = os.getenv(
            "DEEPINFRA_OPENAI_ENDPOINT",
            "https://api.deepinfra.com/v1/openai",
        )
        self.temperature = temperature
        self.response_format = response_format
        self.timeout = float(os.getenv("DEEPINFRA_TIMEOUT", "60"))
        self.max_file_size_mb = 25

    def transcribe_audio(self, audio_file: str, **kwargs):
        """
        Transcribes an audio file using DeepInfra's native speech API.
        """
        self._validate_audio_file(audio_file)
        self._validate_config()

        data = self._build_request_data(kwargs)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        with open(audio_file, "rb") as f:
            files = {"audio": f}
            response = requests.post(
                self._transcription_url(kwargs.get("model")),
                headers=headers,
                data=data,
                files=files,
                timeout=kwargs.get("timeout", self.timeout),
            )

        if response.status_code != 200:
            raise Exception(f"DeepInfra transcription failed: {response.text}")

        try:
            return response.json()
        except ValueError:
            return response.text

    def generate_corrected_transcript(
        self,
        audio_file: str,
        temperature: float,
        system_prompt: str,
    ):
        """
        Uses DeepInfra's OpenAI-compatible chat API to correct transcript text.
        """
        self._validate_config()
        if not self.model_name_chat:
            raise ValueError(
                "DEEPINFRA_MODEL_NAME_CHAT is required when using --correct "
                "with the DeepInfra provider."
            )

        transcription = self.transcribe_audio(audio_file)
        transcription_text = (
            transcription.get("text", "")
            if isinstance(transcription, dict)
            else transcription
        )

        client = OpenAI(api_key=self.api_key, base_url=self.openai_endpoint)
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

    def _transcription_url(self, model_override=None):
        if self.endpoint:
            return self.endpoint

        model = model_override or self.model
        return f"https://api.deepinfra.com/v1/inference/{model}"

    def _build_request_data(self, kwargs):
        data = {}
        for key in ("language", "prompt", "task"):
            value = kwargs.get(key)
            if value:
                data[key] = value

        temperature = kwargs.get("temperature", self.temperature)
        if temperature is not None:
            data["temperature"] = temperature

        return data

    def _validate_config(self):
        if not self.api_key:
            raise ValueError("DEEPINFRA_TOKEN is required for DeepInfra transcription.")

    def _validate_audio_file(self, audio_file: str):
        if not os.path.exists(audio_file):
            raise ValueError(f"File {audio_file} does not exist.")

        file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            raise Exception(
                f"File size exceeds the maximum limit of {self.max_file_size_mb} MB."
            )

        valid_extensions = [".mp3", ".wav"]
        if not any(str(audio_file).endswith(ext) for ext in valid_extensions):
            raise ValueError(
                f"Unsupported audio file format: {audio_file}. "
                f"Supported formats are {valid_extensions}."
            )
