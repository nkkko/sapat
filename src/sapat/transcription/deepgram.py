import os
import requests
from dotenv import load_dotenv
from openai import OpenAI
from .base import TranscriptionBase

# Load environment variables
load_dotenv(".env")


class DeepgramTranscription(TranscriptionBase):
    """
    Deepgram API implementation for transcription.
    """

    def __init__(self, temperature: float, response_format: str = "json"):
        """
        Initializes the DeepgramTranscription class.

        Parameters:
        - temperature (float): Default temperature value for transcript correction.
        - response_format (str): Default response format for transcription.
        """
        self.api_key = os.getenv("DEEPGRAM_API_KEY")
        self.model = os.getenv("DEEPGRAM_MODEL", "nova-3")
        self.endpoint = os.getenv("DEEPGRAM_API_ENDPOINT", "https://api.deepgram.com/v1/listen")
        self.correction_api_key = os.getenv("DEEPGRAM_CORRECTION_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.correction_model = (
            os.getenv("DEEPGRAM_CORRECTION_MODEL")
            or os.getenv("OPENAI_MODEL_NAME_CHAT")
            or "gpt-4o-mini"
        )
        self.temperature = temperature
        self.response_format = response_format
        self.max_file_size_mb = 500

    def transcribe_audio(self, audio_file: str, **kwargs):
        """
        Transcribes an audio file using Deepgram's pre-recorded audio API.

        Parameters:
        - audio_file (str): Path to the audio file.
        - kwargs: Additional parameters for transcription.

        Returns:
        - dict or str: The transcription result.
        """
        self._validate_audio_file(audio_file)
        if not self.api_key:
            raise ValueError("DEEPGRAM_API_KEY is required when using --api deepgram.")

        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": self._content_type_for(audio_file),
        }
        params = {
            "model": kwargs.get("model", self.model),
            "smart_format": "true",
            "punctuate": "true",
        }

        language = kwargs.get("language")
        if language:
            params["language"] = language

        with open(audio_file, "rb") as f:
            response = requests.post(self.endpoint, headers=headers, params=params, data=f)

        if response.status_code != 200:
            raise Exception(f"Transcription failed: {response.text}")

        payload = response.json()
        transcript = self._extract_transcript(payload)
        if kwargs.get("response_format", self.response_format) == "text":
            return transcript
        return {"text": transcript, "raw": payload}

    def generate_corrected_transcript(self, audio_file: str, temperature: float, system_prompt: str):
        """
        Uses an OpenAI-compatible chat model to correct Deepgram transcript text.

        Parameters:
        - audio_file (str): Path to the audio file for transcription.
        - temperature (float): The sampling temperature for the chat API.
        - system_prompt (str): The system prompt to guide the assistant.

        Returns:
        - str: The corrected transcription.
        """
        if not self.correction_api_key:
            raise ValueError(
                "OPENAI_API_KEY or DEEPGRAM_CORRECTION_OPENAI_API_KEY is required "
                "when using --correct with --api deepgram."
            )

        transcription = self.transcribe_audio(audio_file)
        transcription_text = transcription.get("text", "") if isinstance(transcription, dict) else transcription

        client = OpenAI(api_key=self.correction_api_key)
        response = client.chat.completions.create(
            model=self.correction_model,
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
            raise ValueError(f"Unsupported audio file format: {audio_file}. Supported formats are {valid_extensions}.")

    def _content_type_for(self, audio_file: str):
        if str(audio_file).endswith(".mp3"):
            return "audio/mpeg"
        if str(audio_file).endswith(".wav"):
            return "audio/wav"
        if str(audio_file).endswith(".flac"):
            return "audio/flac"
        return "application/octet-stream"

    def _extract_transcript(self, payload: dict):
        try:
            return payload["results"]["channels"][0]["alternatives"][0].get("transcript", "")
        except (KeyError, IndexError, TypeError) as exc:
            raise Exception("Deepgram response did not include a transcript.") from exc
