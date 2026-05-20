import os

import fal_client
from dotenv import load_dotenv

from .base import TranscriptionBase

# Load environment variables
load_dotenv(".env")


class FalTranscription(TranscriptionBase):
    """
    fal.ai Whisper API implementation for transcription.
    """

    def __init__(self, temperature: float):
        """
        Initializes the FalTranscription class.

        Parameters:
        - temperature (float): Accepted for CLI compatibility.
        """
        self.api_key = os.getenv("FAL_KEY")
        self.model_id = os.getenv("FAL_MODEL_ID", "fal-ai/whisper")
        self.task = os.getenv("FAL_TASK", "transcribe")
        self.chunk_level = os.getenv("FAL_CHUNK_LEVEL", "segment")
        self.batch_size = int(os.getenv("FAL_BATCH_SIZE", "64"))
        self.temperature = temperature
        self.max_file_size_mb = 25

    def transcribe_audio(self, audio_file: str, **kwargs):
        """
        Transcribes an audio file using fal.ai's Whisper API.

        Parameters:
        - audio_file (str): Path to the audio file.
        - kwargs: Additional parameters for transcription.

        Returns:
        - dict: The fal.ai transcription response.
        """
        self._validate_audio_file(audio_file)

        audio_url = fal_client.upload_file(audio_file)
        arguments = {
            "audio_url": audio_url,
            "task": kwargs.get("task", self.task),
            "chunk_level": kwargs.get("chunk_level", self.chunk_level),
            "batch_size": kwargs.get("batch_size", self.batch_size),
        }

        language = kwargs.get("language")
        if language:
            arguments["language"] = language

        prompt = kwargs.get("prompt")
        if prompt:
            arguments["prompt"] = prompt

        return fal_client.subscribe(self.model_id, arguments=arguments)

    def generate_corrected_transcript(self, audio_file: str, temperature: float, system_prompt: str):
        """
        Transcript correction is not implemented for fal.ai.
        """
        raise NotImplementedError("Correction is not implemented for the fal.ai transcription API.")

    def _validate_audio_file(self, audio_file: str):
        """
        Validates the audio file for size, format, and configuration.
        """
        if not self.api_key:
            raise ValueError("FAL_KEY is required for fal.ai transcription.")

        if not os.path.exists(audio_file):
            raise ValueError(f"File {audio_file} does not exist.")

        file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            raise Exception(f"File size exceeds the maximum limit of {self.max_file_size_mb} MB.")

        valid_extensions = [".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"]
        if not any(str(audio_file).endswith(ext) for ext in valid_extensions):
            raise ValueError(f"Unsupported audio file format: {audio_file}. Supported formats are {valid_extensions}.")
