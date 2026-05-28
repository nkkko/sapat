# ABOUTME: Gladia pre-recorded transcription provider
# ABOUTME: Uploads audio, creates transcription job, polls for completion

import os
from typing import Optional

import requests

from sapat.providers import register
from sapat.providers.async_poll import AsyncPollProvider
from sapat.providers.base import (
    ProviderConfig,
    TranscriptionResult,
)


@register
class GladiaProvider(AsyncPollProvider):
    """Gladia v2 pre-recorded transcription provider."""

    name = "gladia"
    config = ProviderConfig(
        required_env_vars=["GLADIA_API_KEY"],
        default_model="default",
    )

    poll_interval: float = float(os.getenv("GLADIA_POLL_INTERVAL_SECONDS", "2"))
    max_poll_time: float = float(os.getenv("GLADIA_TIMEOUT_SECONDS", "600"))

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("GLADIA_API_KEY")
        self.upload_endpoint = os.getenv(
            "GLADIA_UPLOAD_ENDPOINT", "https://api.gladia.io/v2/upload"
        )
        self.transcription_endpoint = os.getenv(
            "GLADIA_TRANSCRIPTION_ENDPOINT",
            "https://api.gladia.io/v2/pre-recorded",
        )
        # Stashed during _upload for use in _fetch_result
        self._result_url: Optional[str] = None

    def _headers(self) -> dict:
        return {"x-gladia-key": self.api_key}

    def _upload(self, audio_file: str, model: str, language: str, **kwargs) -> str:
        audio_url = self._upload_audio(audio_file)
        result_url = self._create_transcription(audio_url, language)
        self._result_url = result_url
        # Use the result_url as the job_id for polling
        return result_url

    def _upload_audio(self, audio_file: str) -> str:
        with open(audio_file, "rb") as f:
            response = requests.post(
                self.upload_endpoint,
                headers=self._headers(),
                files={"audio": f},
                timeout=60,
            )
        if response.status_code not in (200, 201):
            raise RuntimeError(f"Gladia upload failed: {response.text}")

        audio_url = response.json().get("audio_url")
        if not audio_url:
            raise RuntimeError("Gladia upload response did not include audio_url.")
        return audio_url

    def _create_transcription(self, audio_url: str, language: str) -> str:
        payload: dict = {"audio_url": audio_url}
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
            raise RuntimeError(f"Gladia transcription job failed: {response.text}")

        job = response.json()
        result_url = job.get("result_url")
        if result_url:
            return result_url

        job_id = job.get("id")
        if not job_id:
            raise RuntimeError(
                "Gladia job response did not include result_url or id."
            )
        return f"{self.transcription_endpoint}/{job_id}"

    def _poll(self, job_id: str) -> str:
        response = requests.get(job_id, headers=self._headers(), timeout=60)
        if response.status_code != 200:
            raise RuntimeError(f"Gladia polling failed: {response.text}")

        result = response.json()
        status = result.get("status")
        if status == "done":
            return "completed"
        if status in {"error", "failed"}:
            return "failed"
        return "pending"

    def _fetch_result(self, job_id: str) -> TranscriptionResult:
        response = requests.get(job_id, headers=self._headers(), timeout=60)
        if response.status_code != 200:
            raise RuntimeError(f"Gladia result fetch failed: {response.text}")

        result = response.json()
        text = (
            result.get("result", {})
            .get("transcription", {})
            .get("full_transcript", "")
        )
        return TranscriptionResult(text=text)
