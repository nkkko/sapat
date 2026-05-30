# ABOUTME: Rev AI asynchronous transcription provider
# ABOUTME: Submits local audio, polls job status, and fetches plain-text transcripts

import os

import requests

from sapat.providers import register
from sapat.providers.async_poll import AsyncPollProvider
from sapat.providers.base import (
    AudioFormat,
    ProviderConfig,
    TranscriptionResult,
)


@register
class RevAIProvider(AsyncPollProvider):
    """Rev AI asynchronous Speech-to-Text API provider."""

    name = "revai"
    config = ProviderConfig(
        required_env_vars=["REVAI_ACCESS_TOKEN"],
        max_file_size_mb=2048.0,
        preferred_format=AudioFormat.MP3,
        supports_correction=False,
        default_model="async",
    )

    def __init__(self):
        super().__init__()
        self.access_token = os.getenv("REVAI_ACCESS_TOKEN", "")
        self.base_url = os.getenv(
            "REVAI_API_BASE_URL",
            "https://api.rev.ai/speechtotext/v1",
        ).rstrip("/")
        self.poll_interval = float(os.getenv("REVAI_JOB_POLL_INTERVAL_SECONDS", "5"))
        self.max_poll_time = float(os.getenv("REVAI_JOB_TIMEOUT_SECONDS", "3600"))

    def _headers(self, accept: str = "application/json") -> dict:
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": accept,
        }

    def _upload(self, audio_file: str, model: str, language: str, **kwargs) -> str:
        data = {}
        if language and language.lower() != "auto":
            data["language"] = language

        with open(audio_file, "rb") as f:
            response = requests.post(
                f"{self.base_url}/jobs",
                headers=self._headers(),
                data=data,
                files={"media": (os.path.basename(audio_file), f)},
                timeout=120,
            )

        if response.status_code not in (200, 201):
            raise RuntimeError(f"Rev AI job creation failed: {response.text}")

        job_id = response.json().get("id")
        if not job_id:
            raise RuntimeError("Rev AI job creation response did not include an id.")
        return job_id

    def _poll(self, job_id: str) -> str:
        response = requests.get(
            f"{self.base_url}/jobs/{job_id}",
            headers=self._headers(),
            timeout=30,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Rev AI status check failed: {response.text}")

        status = response.json().get("status")
        if status == "transcribed":
            return "completed"
        if status == "failed":
            return "failed"
        return "pending"

    def _fetch_result(self, job_id: str) -> TranscriptionResult:
        response = requests.get(
            f"{self.base_url}/jobs/{job_id}/transcript",
            headers=self._headers("text/plain"),
            timeout=60,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Rev AI transcript retrieval failed: {response.text}")

        return TranscriptionResult(text=response.text, raw_response=response.text)
