import os

import requests
from dotenv import load_dotenv

from .base import TranscriptionBase


load_dotenv(".env")


class NanoGPTTranscription(TranscriptionBase):
    """
    NanoGPT OpenAI-compatible API implementation for transcription.
    """

    def __init__(self, temperature: float, response_format: str = "json"):
        self.api_key = os.getenv("NANOGPT_API_KEY")
        self.model = os.getenv("NANOGPT_MODEL", "Whisper-Large-V3")
        self.endpoint = os.getenv(
            "NANOGPT_API_ENDPOINT",
            "https://nano-gpt.com/api/v1/audio/transcriptions",
        )
        self.chat_model = os.getenv("NANOGPT_CHAT_MODEL")
        self.chat_endpoint = os.getenv(
            "NANOGPT_CHAT_ENDPOINT",
            "https://nano-gpt.com/api/v1/chat/completions",
        )
        self.temperature = temperature
        self.response_format = response_format
        self.max_file_size_mb = 25

    def transcribe_audio(self, audio_file: str, **kwargs):
        """
        Transcribes an audio file using NanoGPT's OpenAI-compatible STT API.

        Parameters:
        - audio_file (str): Path to the audio file.
        - kwargs: Additional parameters for transcription.

        Returns:
        - dict or str: The transcription result.
        """
        self._validate_configuration()
        self._validate_audio_file(audio_file)

        response_format = kwargs.get("response_format", self.response_format)
        data = {
            "model": kwargs.get("model", self.model),
            "response_format": response_format,
            "temperature": kwargs.get("temperature", self.temperature),
        }

        language = kwargs.get("language")
        prompt = kwargs.get("prompt")
        if language:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt

        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        with open(audio_file, "rb") as f:
            response = requests.post(
                self.endpoint,
                headers=headers,
                data=data,
                files={"file": f},
            )

        if response.status_code == 200:
            if response_format in ["json", "verbose_json"]:
                return response.json()
            return response.text

        raise Exception(f"Transcription failed: {response.text}")

    def generate_corrected_transcript(
        self, audio_file: str, temperature: float, system_prompt: str
    ):
        """
        Uses NanoGPT's OpenAI-compatible chat endpoint to correct a transcript.

        Parameters:
        - audio_file (str): Path to the audio file for transcription.
        - temperature (float): The sampling temperature for the chat API.
        - system_prompt (str): The system prompt to guide the assistant.

        Returns:
        - str: The corrected transcription.
        """
        self._validate_configuration()
        if not self.chat_model:
            raise ValueError(
                "NANOGPT_CHAT_MODEL must be set to use --correct with NanoGPT."
            )

        transcription = self.transcribe_audio(audio_file)
        transcription_text = (
            transcription.get("text", "") if isinstance(transcription, dict)
            else transcription
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.chat_model,
            "temperature": temperature,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": transcription_text,
                },
            ],
        }

        response = requests.post(
            self.chat_endpoint,
            headers=headers,
            json=payload,
        )
        if response.status_code != 200:
            raise Exception(f"Transcript correction failed: {response.text}")

        body = response.json()
        return body["choices"][0]["message"]["content"]

    def _validate_configuration(self):
        if not self.api_key:
            raise ValueError("NANOGPT_API_KEY must be set to use the NanoGPT API.")
        if not self.endpoint:
            raise ValueError("NANOGPT_API_ENDPOINT must be set to use the NanoGPT API.")
        if not self.model:
            raise ValueError("NANOGPT_MODEL must be set to use the NanoGPT API.")

    def _validate_audio_file(self, audio_file: str):
        if not os.path.exists(audio_file):
            raise ValueError(f"File {audio_file} does not exist.")

        file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            raise Exception(
                f"File size exceeds the maximum limit of {self.max_file_size_mb} MB."
            )

        valid_extensions = [".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"]
        if not any(str(audio_file).lower().endswith(ext) for ext in valid_extensions):
            raise ValueError(
                f"Unsupported audio file format: {audio_file}. "
                f"Supported formats are {valid_extensions}."
            )
