import os
import requests
from dotenv import load_dotenv
from .base import TranscriptionBase

# Load environment variables
load_dotenv(".env")


class MistralTranscription(TranscriptionBase):
    """
    Mistral API implementation for transcription.
    """

    def __init__(self, temperature: float, response_format: str = "json"):
        """
        Initializes the MistralTranscription class.

        Parameters:
        - temperature (float): Default temperature value for transcription.
        - response_format (str): Default response format for transcription.
        """
        self.api_key = os.getenv("MISTRAL_API_KEY")
        self.model = os.getenv("MISTRAL_MODEL", "voxtral-mini-latest")
        self.endpoint = os.getenv(
            "MISTRAL_API_ENDPOINT",
            "https://api.mistral.ai/v1/audio/transcriptions",
        )
        self.model_name_chat = os.getenv("MISTRAL_MODEL_NAME_CHAT", "mistral-small-latest")
        self.chat_endpoint = os.getenv(
            "MISTRAL_CHAT_API_ENDPOINT",
            "https://api.mistral.ai/v1/chat/completions",
        )
        self.temperature = temperature
        self.response_format = response_format
        self.max_file_size_mb = 25

    def transcribe_audio(self, audio_file: str, **kwargs):
        """
        Transcribes an audio file using Mistral's audio transcription API.

        Parameters:
        - audio_file (str): Path to the audio file.
        - kwargs: Additional parameters for transcription.

        Returns:
        - dict or str: The transcription result.
        """
        self._validate_audio_file(audio_file)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        data = {
            "model": kwargs.get("model", self.model),
            "temperature": kwargs.get("temperature", self.temperature),
        }

        if "language" in kwargs:
            data["language"] = kwargs["language"]

        # Mistral exposes context_bias instead of OpenAI's prompt parameter.
        if kwargs.get("prompt"):
            data["context_bias"] = kwargs["prompt"]

        with open(audio_file, "rb") as f:
            files = {"file": f}
            response = requests.post(self.endpoint, headers=headers, data=data, files=files)

        if response.status_code == 200:
            result = response.json()
            if self.response_format in ["json", "verbose_json"]:
                return result
            return result.get("text", "")
        raise Exception(f"Transcription failed: {response.text}")

    def generate_corrected_transcript(self, audio_file: str, temperature: float, system_prompt: str):
        """
        Uses Mistral's chat API to correct the transcription text.

        Parameters:
        - audio_file (str): Path to the audio file for transcription.
        - temperature (float): The sampling temperature for the chat API.
        - system_prompt (str): The system prompt to guide the assistant.

        Returns:
        - str: The corrected transcription.
        """
        transcription = self.transcribe_audio(audio_file)
        transcription_text = transcription.get("text", "") if isinstance(transcription, dict) else transcription

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name_chat,
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
        response = requests.post(self.chat_endpoint, headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        raise Exception(f"Transcript correction failed: {response.text}")

    def _validate_audio_file(self, audio_file: str):
        """
        Validates the audio file for size and format.

        Parameters:
        - audio_file (str): Path to the audio file.

        Raises:
        - Exception: If the file is invalid.
        """
        if not os.path.exists(audio_file):
            raise ValueError(f"File {audio_file} does not exist.")

        file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            raise Exception(f"File size exceeds the maximum limit of {self.max_file_size_mb} MB.")

        valid_extensions = [".mp3", ".wav", ".flac"]
        if not any(str(audio_file).endswith(ext) for ext in valid_extensions):
            raise ValueError(
                f"Unsupported audio file format: {audio_file}. Supported formats are {valid_extensions}."
            )
