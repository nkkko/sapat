# ABOUTME: AssemblyAI pre-recorded transcription provider
# ABOUTME: Uses upload -> transcript submission -> polling REST workflow

import os
from typing import List, Optional

import requests

from sapat.providers import register
from sapat.providers.async_poll import AsyncPollProvider
from sapat.providers.base import (
    AudioFormat,
    ProviderConfig,
    TranscriptionResult,
)


def _env_bool(name: str) -> Optional[bool]:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    return value.lower() in {"1", "true", "yes", "on"}


@register
class AssemblyAIProvider(AsyncPollProvider):
    """AssemblyAI pre-recorded speech-to-text provider."""

    name = "assemblyai"
    config = ProviderConfig(
        required_env_vars=["ASSEMBLYAI_API_KEY"],
        max_file_size_mb=100.0,
        preferred_format=AudioFormat.MP3,
        default_model="universal-3-pro,universal-2",
    )

    poll_interval: float = float(os.getenv("ASSEMBLYAI_POLL_INTERVAL_SECONDS", "3"))
    max_poll_time: float = float(os.getenv("ASSEMBLYAI_TIMEOUT_SECONDS", "600"))

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("ASSEMBLYAI_API_KEY", "")
        self.base_url = os.getenv(
            "ASSEMBLYAI_BASE_URL", "https://api.assemblyai.com"
        ).rstrip("/")

    def _headers(self, content_type: Optional[str] = None) -> dict:
        headers = {"Authorization": self.api_key}
        if content_type:
            headers["Content-Type"] = content_type
        return headers

    def _speech_models(self, model: str) -> List[str]:
        models = [part.strip() for part in model.split(",") if part.strip()]
        return models or ["universal-3-pro", "universal-2"]

    def _upload(self, audio_file: str, model: str, language: str, **kwargs) -> str:
        upload_url = self._upload_audio(audio_file)
        return self._submit_transcript(upload_url, model, language, **kwargs)

    def _upload_audio(self, audio_file: str) -> str:
        with open(audio_file, "rb") as f:
            response = requests.post(
                f"{self.base_url}/v2/upload",
                headers=self._headers("application/octet-stream"),
                data=f,
                timeout=120,
            )

        if response.status_code != 200:
            raise RuntimeError(f"AssemblyAI upload failed: {response.text}")

        upload_url = response.json().get("upload_url")
        if not upload_url:
            raise RuntimeError("AssemblyAI upload response did not include upload_url.")
        return upload_url

    def _submit_transcript(
        self, audio_url: str, model: str, language: str, **kwargs
    ) -> str:
        payload = {
            "audio_url": audio_url,
            "speech_models": self._speech_models(model),
        }

        if language and language.lower() not in {"auto", "detect"}:
            payload["language_code"] = language
        else:
            payload["language_detection"] = True

        prompt = kwargs.get("prompt")
        if prompt:
            payload["prompt"] = prompt

        for env_name, field_name in (
            ("ASSEMBLYAI_SPEAKER_LABELS", "speaker_labels"),
            ("ASSEMBLYAI_PUNCTUATE", "punctuate"),
            ("ASSEMBLYAI_FORMAT_TEXT", "format_text"),
            ("ASSEMBLYAI_DISFLUENCIES", "disfluencies"),
        ):
            value = _env_bool(env_name)
            if value is not None:
                payload[field_name] = value

        response = requests.post(
            f"{self.base_url}/v2/transcript",
            headers=self._headers("application/json"),
            json=payload,
            timeout=60,
        )

        if response.status_code not in (200, 201):
            raise RuntimeError(
                f"AssemblyAI transcript submission failed: {response.text}"
            )

        transcript_id = response.json().get("id")
        if not transcript_id:
            raise RuntimeError("AssemblyAI transcript response did not include id.")
        return transcript_id

    def _poll(self, job_id: str) -> str:
        result = self._get_transcript(job_id)
        status = result.get("status")
        if status == "completed":
            return "completed"
        if status in {"error", "failed"}:
            return "failed"
        return "pending"

    def _fetch_result(self, job_id: str) -> TranscriptionResult:
        result = self._get_transcript(job_id)
        if result.get("status") != "completed":
            raise RuntimeError(f"AssemblyAI transcript {job_id} is not completed.")

        return TranscriptionResult(
            text=result.get("text", ""),
            language=result.get("language_code"),
            duration=result.get("audio_duration"),
            segments=result.get("utterances") or result.get("words"),
            raw_response=result,
        )

    def _get_transcript(self, transcript_id: str) -> dict:
        response = requests.get(
            f"{self.base_url}/v2/transcript/{transcript_id}",
            headers=self._headers(),
            timeout=30,
        )
        if response.status_code != 200:
            raise RuntimeError(f"AssemblyAI transcript fetch failed: {response.text}")
        return response.json()
