import os
import time
from typing import Optional

import requests
from dotenv import load_dotenv

from .base import TranscriptionBase


load_dotenv(".env")


class GladiaTranscription(TranscriptionBase):
    """
    Gladia API implementation for pre-recorded transcription.
    """

    def __init__(
        self,
        temperature: float,
        response_format: str = "json",
        poll_interval_seconds: Optional[float] = None,
        timeout_seconds: Optional[float] = None,
    ):
        self.api_key = os.getenv("GLADIA_API_KEY")
        self.upload_endpoint = os.getenv(
            "GLADIA_UPLOAD_ENDPOINT", "https://api.gladia.io/v2/upload"
        )
        self.transcription_endpoint = os.getenv(
            "GLADIA_TRANSCRIPTION_ENDPOINT",
            "https://api.gladia.io/v2/pre-recorded",
        )
        self.temperature = temperature
        self.response_format = response_format
        self.poll_interval_seconds = (
            poll_interval_seconds
            if poll_interval_seconds is not None
            else float(os.getenv("GLADIA_POLL_INTERVAL_SECONDS", "2"))
        )
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else float(os.getenv("GLADIA_TIMEOUT_SECONDS", "600"))
        )

    def transcribe_audio(self, audio_file: str, **kwargs):
        """
        Transcribes an audio file using Gladia's v2 pre-recorded flow.
        """
        self._validate_audio_file(audio_file)
        audio_url = self._upload_audio(audio_file)
        result_url = self._create_transcription(audio_url, kwargs)
        return self._poll_transcription(result_url)

    def _headers(self):
        if not self.api_key:
            raise ValueError("GLADIA_API_KEY is required for Gladia transcription.")
        return {"x-gladia-key": self.api_key}

    def _upload_audio(self, audio_file: str):
        with open(audio_file, "rb") as f:
            response = requests.post(
                self.upload_endpoint,
                headers=self._headers(),
                files={"audio": f},
                timeout=60,
            )
        if response.status_code not in (200, 201):
            raise Exception(f"Gladia upload failed: {response.text}")

        audio_url = response.json().get("audio_url")
        if not audio_url:
            raise Exception("Gladia upload response did not include audio_url.")
        return audio_url

    def _create_transcription(self, audio_url: str, kwargs):
        payload = {
            "audio_url": audio_url,
        }
        language = kwargs.get("language")
        if language:
            payload["language_config"] = {
                "languages": [language],
                "code_switching": False,
            }

        response = requests.post(
            self.transcription_endpoint,
            headers={**self._headers(), "Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        if response.status_code not in (200, 201):
            raise Exception(f"Gladia transcription job failed: {response.text}")

        job = response.json()
        result_url = job.get("result_url")
        if result_url:
            return result_url

        job_id = job.get("id")
        if not job_id:
            raise Exception("Gladia job response did not include result_url or id.")
        return f"{self.transcription_endpoint}/{job_id}"

    def _poll_transcription(self, result_url: str):
        deadline = time.monotonic() + self.timeout_seconds
        last_status = None

        while time.monotonic() < deadline:
            response = requests.get(result_url, headers=self._headers(), timeout=60)
            if response.status_code != 200:
                raise Exception(f"Gladia polling failed: {response.text}")

            result = response.json()
            last_status = result.get("status")
            if last_status == "done":
                return {
                    "text": (
                        result.get("result", {})
                        .get("transcription", {})
                        .get("full_transcript", "")
                    )
                }
            if last_status in {"error", "failed"}:
                raise Exception(f"Gladia transcription failed: {result}")

            time.sleep(self.poll_interval_seconds)

        raise TimeoutError(
            f"Gladia transcription timed out after {self.timeout_seconds} seconds "
            f"(last status: {last_status})."
        )

    def _validate_audio_file(self, audio_file: str):
        if not os.path.exists(audio_file):
            raise ValueError(f"File {audio_file} does not exist.")

        valid_extensions = [".mp3", ".wav", ".flac", ".m4a", ".mp4"]
        if not any(str(audio_file).endswith(ext) for ext in valid_extensions):
            raise ValueError(
                f"Unsupported audio file format: {audio_file}. "
                f"Supported formats are {valid_extensions}."
            )
