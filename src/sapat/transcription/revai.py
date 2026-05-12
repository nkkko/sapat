import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

from .base import TranscriptionBase


load_dotenv(".env")


class RevAITranscription(TranscriptionBase):
    """
    Rev AI asynchronous speech-to-text implementation.
    """

    def __init__(self, temperature: float):
        self.access_token = os.getenv("REVAI_ACCESS_TOKEN")
        if not self.access_token:
            raise ValueError("REVAI_ACCESS_TOKEN is required to use Rev AI.")

        self.endpoint = os.getenv(
            "REVAI_API_ENDPOINT",
            "https://api.rev.ai/speechtotext/v1",
        ).rstrip("/")
        self.poll_interval_seconds = float(
            os.getenv("REVAI_POLL_INTERVAL_SECONDS", "5")
        )
        self.timeout_seconds = float(os.getenv("REVAI_TIMEOUT_SECONDS", "900"))
        self.temperature = temperature

    def transcribe_audio(self, audio_file: str, **kwargs):
        self._validate_audio_file(audio_file)
        job_id = self._submit_job(audio_file, **kwargs)
        self._wait_for_transcription(job_id)
        return self._download_transcript(job_id)

    def _submit_job(self, audio_file: str, **kwargs):
        headers = self._auth_headers()
        options = {"transcriber": "machine"}

        language = kwargs.get("language")
        if language:
            options["language"] = language

        prompt = kwargs.get("prompt")
        if prompt:
            options["metadata"] = prompt

        with open(audio_file, "rb") as media:
            response = requests.post(
                f"{self.endpoint}/jobs",
                headers=headers,
                data={"options": json.dumps(options)},
                files={"media": media},
            )

        if response.status_code not in (200, 201):
            raise Exception(f"Rev AI job submission failed: {response.text}")

        payload = response.json()
        job_id = payload.get("id")
        if not job_id:
            raise Exception(f"Rev AI did not return a job id: {payload}")
        return job_id

    def _wait_for_transcription(self, job_id: str):
        deadline = time.monotonic() + self.timeout_seconds

        while time.monotonic() <= deadline:
            response = requests.get(
                f"{self.endpoint}/jobs/{job_id}",
                headers=self._auth_headers(),
            )
            if response.status_code != 200:
                raise Exception(f"Rev AI job status check failed: {response.text}")

            payload = response.json()
            status = payload.get("status")
            if status == "transcribed":
                return
            if status == "failed":
                failure = payload.get("failure") or payload.get("failure_detail")
                raise Exception(f"Rev AI transcription failed: {failure}")

            time.sleep(self.poll_interval_seconds)

        raise TimeoutError(
            f"Rev AI transcription did not complete within {self.timeout_seconds} seconds."
        )

    def _download_transcript(self, job_id: str):
        response = requests.get(
            f"{self.endpoint}/jobs/{job_id}/transcript",
            headers={**self._auth_headers(), "Accept": "text/plain"},
        )
        if response.status_code != 200:
            raise Exception(f"Rev AI transcript download failed: {response.text}")
        return response.text

    def _auth_headers(self):
        return {"Authorization": f"Bearer {self.access_token}"}

    @staticmethod
    def _validate_audio_file(audio_file: str):
        path = Path(audio_file)
        if not path.exists():
            raise ValueError(f"File {audio_file} does not exist.")

        valid_extensions = [".mp3", ".wav", ".flac", ".m4a"]
        if path.suffix.lower() not in valid_extensions:
            raise ValueError(
                f"Unsupported audio file format: {audio_file}. "
                f"Supported formats are {valid_extensions}."
            )

    @staticmethod
    def generate_corrected_transcript(audio_file, temperature, prompt):
        raise NotImplementedError(
            "Transcript correction is not implemented for Rev AI. "
            "Run without --correct or use an OpenAI-compatible provider."
        )
