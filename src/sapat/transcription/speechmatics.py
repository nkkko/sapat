import json
import os
import time

import requests
from dotenv import load_dotenv

from .base import TranscriptionBase


load_dotenv(".env")


class SpeechmaticsTranscription(TranscriptionBase):
    """
    Speechmatics Batch API implementation for transcription.
    """

    TERMINAL_STATUSES = {"done", "rejected", "deleted", "expired"}

    def __init__(
        self,
        temperature: float,
        poll_interval_seconds: float = 5,
        max_poll_seconds: float = 1800,
    ):
        self.api_key = os.getenv("SPEECHMATICS_API_KEY")
        self.endpoint = os.getenv(
            "SPEECHMATICS_API_ENDPOINT",
            "https://asr.api.speechmatics.com/v2/jobs",
        ).rstrip("/")
        self.operating_point = os.getenv("SPEECHMATICS_OPERATING_POINT", "standard")
        self.temperature = temperature
        self.poll_interval_seconds = poll_interval_seconds
        self.max_poll_seconds = max_poll_seconds
        self.max_file_size_mb = 2048

    def transcribe_audio(self, audio_file: str, **kwargs):
        """
        Transcribes an audio file using the Speechmatics Batch API.
        """
        self._validate_audio_file(audio_file)
        job_id = self._create_job(audio_file, kwargs)
        self._wait_for_job(job_id)
        return self._get_transcript(job_id)

    def _create_job(self, audio_file: str, options: dict) -> str:
        config = {
            "type": "transcription",
            "transcription_config": {
                "language": options.get("language") or "en",
            },
        }

        if self.operating_point:
            config["transcription_config"]["operating_point"] = self.operating_point

        with open(audio_file, "rb") as f:
            response = requests.post(
                self.endpoint,
                headers=self._headers(),
                data={"config": json.dumps(config)},
                files={"data_file": (os.path.basename(audio_file), f)},
            )

        if response.status_code != 201:
            raise Exception(f"Speechmatics job creation failed: {response.text}")

        data = response.json()
        job_id = data.get("id") or data.get("job", {}).get("id")
        if not job_id:
            raise Exception("Speechmatics job creation response did not include a job id.")
        return job_id

    def _wait_for_job(self, job_id: str):
        deadline = time.monotonic() + self.max_poll_seconds

        while time.monotonic() < deadline:
            response = requests.get(f"{self.endpoint}/{job_id}", headers=self._headers())
            if response.status_code != 200:
                raise Exception(f"Speechmatics status check failed: {response.text}")

            status = self._extract_status(response.json())
            if status in self.TERMINAL_STATUSES:
                if status == "done":
                    return
                raise Exception(f"Speechmatics job ended with status: {status}")

            time.sleep(self.poll_interval_seconds)

        raise TimeoutError(f"Timed out waiting for Speechmatics job {job_id}")

    def _get_transcript(self, job_id: str) -> str:
        response = requests.get(
            f"{self.endpoint}/{job_id}/transcript",
            headers=self._headers(),
            params={"format": "txt"},
        )
        if response.status_code != 200:
            raise Exception(f"Speechmatics transcript retrieval failed: {response.text}")
        return response.text

    def _headers(self):
        if not self.api_key:
            raise ValueError("SPEECHMATICS_API_KEY is required for Speechmatics transcription.")
        return {"Authorization": f"Bearer {self.api_key}"}

    @staticmethod
    def _extract_status(data: dict):
        return data.get("status") or data.get("job", {}).get("status")

    def _validate_audio_file(self, audio_file: str):
        if not os.path.exists(audio_file):
            raise ValueError(f"File {audio_file} does not exist.")

        file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            raise Exception(
                f"File size exceeds the maximum limit of {self.max_file_size_mb} MB."
            )

        valid_extensions = [".mp3", ".wav", ".flac", ".m4a", ".mp4"]
        if not any(str(audio_file).endswith(ext) for ext in valid_extensions):
            raise ValueError(
                f"Unsupported audio file format: {audio_file}. "
                f"Supported formats are {valid_extensions}."
            )
