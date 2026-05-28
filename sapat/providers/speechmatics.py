# ABOUTME: Speechmatics Batch API v2 transcription provider
# ABOUTME: Creates a transcription job, polls until done, fetches transcript text

import json
import os

import requests

from sapat.providers import register
from sapat.providers.async_poll import AsyncPollProvider
from sapat.providers.base import (
    ProviderConfig,
    TranscriptionResult,
)


TERMINAL_STATUSES = {"done", "rejected", "deleted", "expired"}


@register
class SpeechmaticsProvider(AsyncPollProvider):
    """Speechmatics Batch API v2 transcription provider."""

    name = "speechmatics"
    config = ProviderConfig(
        required_env_vars=["SPEECHMATICS_API_KEY"],
        default_model="default",
    )

    poll_interval: float = 5.0
    max_poll_time: float = 1800.0

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("SPEECHMATICS_API_KEY")
        self.endpoint = os.getenv(
            "SPEECHMATICS_API_ENDPOINT",
            "https://asr.api.speechmatics.com/v2/jobs",
        ).rstrip("/")
        self.operating_point = os.getenv("SPEECHMATICS_OPERATING_POINT", "standard")

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _upload(self, audio_file: str, model: str, language: str, **kwargs) -> str:
        config = {
            "type": "transcription",
            "transcription_config": {
                "language": language,
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
            raise RuntimeError(f"Speechmatics job creation failed: {response.text}")

        data = response.json()
        job_id = data.get("id") or data.get("job", {}).get("id")
        if not job_id:
            raise RuntimeError(
                "Speechmatics job creation response did not include a job id."
            )
        return job_id

    def _poll(self, job_id: str) -> str:
        response = requests.get(
            f"{self.endpoint}/{job_id}", headers=self._headers()
        )
        if response.status_code != 200:
            raise RuntimeError(f"Speechmatics status check failed: {response.text}")

        status = self._extract_status(response.json())
        if status == "done":
            return "completed"
        if status in TERMINAL_STATUSES:
            return "failed"
        return "pending"

    def _fetch_result(self, job_id: str) -> TranscriptionResult:
        response = requests.get(
            f"{self.endpoint}/{job_id}/transcript",
            headers=self._headers(),
            params={"format": "txt"},
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Speechmatics transcript retrieval failed: {response.text}"
            )
        return TranscriptionResult(text=response.text)

    @staticmethod
    def _extract_status(data: dict) -> str:
        return data.get("status") or data.get("job", {}).get("status")
