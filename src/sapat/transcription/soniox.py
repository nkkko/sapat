import os

from dotenv import load_dotenv
from soniox import SonioxClient

from .base import TranscriptionBase


load_dotenv(".env")


class SonioxTranscription(TranscriptionBase):
    """
    Soniox API implementation for asynchronous file transcription.
    """

    def __init__(self, temperature: float, response_format: str = "json"):
        self.api_key = os.getenv("SONIOX_API_KEY")
        self.model = os.getenv("SONIOX_MODEL", "stt-async-v4")
        self.destroy_after_transcription = os.getenv(
            "SONIOX_DESTROY_AFTER_TRANSCRIPTION", "true"
        ).lower() not in {"0", "false", "no"}
        self.temperature = temperature
        self.response_format = response_format
        self.client = SonioxClient()

    def transcribe_audio(self, audio_file: str, **kwargs):
        """
        Transcribes an audio file using Soniox asynchronous Speech-to-Text.

        Parameters:
        - audio_file (str): Path to the audio file.
        - kwargs: Additional arguments for transcription.

        Returns:
        - dict: A dictionary containing the transcript text.
        """
        self._validate_audio_file(audio_file)

        transcription = self.client.stt.transcribe(
            model=kwargs.get("model", self.model),
            file=audio_file,
        )
        self.client.stt.wait(transcription.id)
        transcript = self.client.stt.get_transcript(transcription.id)

        if self.destroy_after_transcription:
            self.client.stt.destroy(transcription.id)

        return {"text": transcript.text}

    def generate_corrected_transcript(self, audio_file: str, temperature: float, system_prompt: str):
        raise NotImplementedError(
            "Transcript correction is not implemented for the Soniox provider yet."
        )

    def _validate_audio_file(self, audio_file: str):
        if not os.path.exists(audio_file):
            raise ValueError(f"File {audio_file} does not exist.")

        valid_extensions = [".mp3", ".wav", ".flac", ".m4a", ".mp4", ".webm"]
        if not any(str(audio_file).lower().endswith(ext) for ext in valid_extensions):
            raise ValueError(
                f"Unsupported audio file format: {audio_file}. "
                f"Supported formats are {valid_extensions}."
            )
