import os
import requests
from openai import OpenAI
from dotenv import load_dotenv
from .base import TranscriptionBase

# Load environment variables
load_dotenv(".env")

class OpenAITranscription(TranscriptionBase):
    """
    OpenAI API implementation for transcription.
    """

    def __init__(self, temperature: float, response_format: str = "json"):
        """
        Initializes the OpenAITranscription class.

        Parameters:
        - temperature (float): Default temperature value for transcription.
        - response_format (str): Default response format for transcription.
        """
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.model = os.getenv('OPENAI_MODEL')
        self.endpoint = os.getenv('OPENAI_API_ENDPOINT')
        self.model_name_chat = os.getenv('OPENAI_MODEL_NAME_CHAT')
        self.temperature = temperature
        self.response_format = response_format
        self.max_file_size_mb = 25


    def transcribe_audio(self, audio_file: str, **kwargs):
        """
        Transcribes an audio file using OpenAI Whisper API.

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
            "response_format": kwargs.get("response_format", self.response_format),
            "temperature": kwargs.get("temperature", self.temperature),
        }

        if "language" in kwargs:
            data["language"] = kwargs["language"]
        if "prompt" in kwargs:
            data["prompt"] = kwargs["prompt"]

        with open(audio_file, 'rb') as f:
            files = {'file': f}
            response = requests.post(self.endpoint, headers=headers, data=data, files=files)

        if response.status_code == 200:
            if data["response_format"] in ["json", "verbose_json"]:
                return response.json()
            return response.text
        else:
            raise Exception(f"Transcription failed: {response.text}")

    def generate_corrected_transcript(self, audio_file: str, temperature: float, system_prompt: str):
        """
        Uses OpenAI's GPT API to correct the transcription text.

        Parameters:
        - audio_file (str): Path to the audio file for transcription.
        - temperature (float): The sampling temperature for the chat API.
        - system_prompt (str): The system prompt to guide the assistant.

        Returns:
        - str: The corrected transcription.
        """
        client = OpenAI(
            api_key=self.api_key
        )

        # First, get the transcription text
        transcription = self.transcribe_audio(audio_file)
        transcription_text = transcription.get('text', '') if isinstance(transcription, dict) else transcription

        # Use the GPT model to correct the transcription
        response = client.chat.completions.create(
            model=self.model_name_chat,
            temperature=temperature,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": transcription_text
                }
            ]
        )
        return response.choices[0].message.content


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

        valid_extensions = ['.mp3', '.wav', '.flac']
        if not any(str(audio_file).endswith(ext) for ext in valid_extensions):
            raise ValueError(f"Unsupported audio file format: {audio_file}. Supported formats are {valid_extensions}.")
