import os
import requests
from openai import AzureOpenAI
from dotenv import load_dotenv
from .base import TranscriptionBase

# Load environment variables
load_dotenv()

class AzureTranscription(TranscriptionBase):
    """
    Azure API implementation for transcription.
    """

    def __init__(self, temperature: float, response_format: str = "json"):
        """
        Initializes the AzureTranscription class.

        Parameters:
        - temperature (float): Default temperature value for transcription.
        - response_format (str): Default response format for transcription.
        """
        self.api_key = os.getenv('AZURE_OPENAI_API_KEY')
        self.endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        self.deployment_name_whisper = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME_WHISPER')
        self.api_version_whisper = os.getenv("AZURE_OPENAI_API_VERSION_WHISPER")
        self.deployment_name_chat = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME_CHAT')
        self.api_version_chat = os.getenv("AZURE_OPENAI_API_VERSION_CHAT")
        self.temperature = temperature
        self.response_format = response_format

    def transcribe_audio(self, audio_file: str, **kwargs):
        """
        Transcribes an audio file using Azure OpenAI Whisper API.

        Parameters:
        - audio_file (str): Path to the audio file.
        - kwargs: Additional parameters for transcription.

        Returns:
        - dict or str: The transcription result.
        """
        url = f"{self.endpoint}/openai/deployments/{self.deployment_name_whisper}/audio/translations?api-version={self.api_version_whisper}"

        headers = {
            "api-key": self.api_key,
        }

        data = {
            "response_format": kwargs.get("response_format", self.response_format),
            "temperature": kwargs.get("temperature", self.temperature)
        }

        if "language" in kwargs:
            data["language"] = kwargs["language"]
        if "prompt" in kwargs:
            data["prompt"] = kwargs["prompt"]
        if "timestamp_granularities" in kwargs:
            data["timestamp_granularities"] = kwargs["timestamp_granularities"]

        with open(audio_file, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, headers=headers, data=data, files=files)

        if response.status_code == 200:
            if data["response_format"] in ["json", "verbose_json"]:
                return response.json()
            return response.text
        else:
            raise Exception(f"Transcription failed: {response.text}")

    def generate_corrected_transcript(self, temperature: float, system_prompt: str, audio_file: str):
        """
        Uses Azure's chat API to correct the transcription text.

        Parameters:
        - temperature (float): The sampling temperature for the chat API.
        - system_prompt (str): The system prompt to guide the assistant.
        - audio_file (str): Path to the audio file for transcription.

        Returns:
        - str: The corrected transcription.
        """
        client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.endpoint,
            api_version=self.api_version_chat
        )

        # First, get the transcription text
        transcription = self.transcribe_audio(audio_file)
        transcription_text = transcription.get('text', '') if isinstance(transcription, dict) else transcription

        # Use the chat API to correct the text
        response = client.chat.completions.create(
            model=self.deployment_name_chat,
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

