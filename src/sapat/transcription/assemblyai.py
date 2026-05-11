import os
import time
from typing import Optional

import requests
from dotenv import load_dotenv

from .base import TranscriptionBase

load_dotenv(".env")


class AssemblyAITranscription(TranscriptionBase):
    """
    AssemblyAI API implementation for transcription.
    """

    def __init__(self, temperature: float, response_format: str = "json"):
        """
        Initializes the AssemblyAITranscription class.

        Parameters:
        - temperature (float): Kept for compatibility with the common interface.
        - response_format (str): Kept for compatibility with the common interface.
        """
        self.api_key = os.getenv("ASSEMBLYAI_API_KEY")
        self.endpoint = os.getenv("ASSEMBLYAI_API_ENDPOINT", "https://api.assemblyai.com/v2")
        self.poll_interval_seconds = float(os.getenv("ASSEMBLYAI_POLL_INTERVAL_SECONDS", "3"))
        self.timeout_seconds = float(os.getenv("ASSEMBLYAI_TIMEOUT_SECONDS", "600"))
        self.temperature = temperature
        self.response_format = response_format

    def transcribe_audio(self, audio_file: str, **kwargs):
        """
        Transcribes an audio file using the AssemblyAI async transcription API.

        Parameters:
        - audio_file (str): Path to the audio file.
        - kwargs: Additional parameters for transcription.

        Returns:
        - dict: The completed AssemblyAI transcript response.
        """
        self._validate_configuration()
        upload_url = self._upload_file(audio_file)
        transcript_id = self._create_transcript(
            upload_url=upload_url,
            language_code=kwargs.get("language"),
        )
        return self._wait_for_transcript(transcript_id)

    def _headers(self):
        return {"authorization": self.api_key}

    def _upload_file(self, audio_file: str):
        with open(audio_file, "rb") as f:
            response = requests.post(
                f"{self.endpoint}/upload",
                headers={**self._headers(), "content-type": "application/octet-stream"},
                data=f,
            )

        if response.status_code != 200:
            raise Exception(f"AssemblyAI upload failed: {response.text}")

        upload_url = response.json().get("upload_url")
        if not upload_url:
            raise Exception("AssemblyAI upload response did not include upload_url")
        return upload_url

    def _create_transcript(self, upload_url: str, language_code: Optional[str] = None):
        payload = {"audio_url": upload_url}
        if language_code:
            payload["language_code"] = language_code

        response = requests.post(
            f"{self.endpoint}/transcript",
            headers={**self._headers(), "content-type": "application/json"},
            json=payload,
        )

        if response.status_code not in (200, 201):
            raise Exception(f"AssemblyAI transcript creation failed: {response.text}")

        transcript_id = response.json().get("id")
        if not transcript_id:
            raise Exception("AssemblyAI transcript response did not include id")
        return transcript_id

    def _wait_for_transcript(self, transcript_id: str):
        deadline = time.monotonic() + self.timeout_seconds

        while time.monotonic() < deadline:
            response = requests.get(
                f"{self.endpoint}/transcript/{transcript_id}",
                headers=self._headers(),
            )

            if response.status_code != 200:
                raise Exception(f"AssemblyAI transcript polling failed: {response.text}")

            payload = response.json()
            status = payload.get("status")
            if status == "completed":
                return payload
            if status == "error":
                raise Exception(f"AssemblyAI transcription failed: {payload.get('error')}")

            time.sleep(self.poll_interval_seconds)

        raise TimeoutError(f"AssemblyAI transcription timed out after {self.timeout_seconds} seconds")

    def _validate_configuration(self):
        if not self.api_key:
            raise ValueError("ASSEMBLYAI_API_KEY must be set in the environment")
