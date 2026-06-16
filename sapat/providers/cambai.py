# ABOUTME: CAMB.AI async transcription provider
# ABOUTME: Uses POST /transcribe, polls task status, then fetches run transcript

import os
from typing import Any, Dict, List, Optional

import requests

from sapat.providers import register
from sapat.providers.async_poll import AsyncPollProvider
from sapat.providers.base import (
    AudioFormat,
    ProviderConfig,
    TranscriptionResult,
)


LANGUAGE_ALIASES = {
    "ar": "ar-sa",
    "de": "de-de",
    "en": "en-us",
    "es": "es-es",
    "fr": "fr-fr",
    "ja": "ja-jp",
    "zh": "zh-cn",
}


@register
class CambAIProvider(AsyncPollProvider):
    """CAMB.AI speech-to-text provider."""

    name = "cambai"
    config = ProviderConfig(
        required_env_vars=["CAMB_API_KEY"],
        max_file_size_mb=20.0,
        preferred_format=AudioFormat.MP3,
        supports_correction=False,
        default_model="default",
    )

    poll_interval: float = float(os.getenv("CAMB_POLL_INTERVAL_SECONDS", "5"))
    max_poll_time: float = float(os.getenv("CAMB_TIMEOUT_SECONDS", "600"))

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("CAMB_API_KEY")
        self.base_url = os.getenv(
            "CAMB_API_BASE_URL", "https://client.camb.ai/apis"
        ).rstrip("/")
        self.word_level_timestamps = os.getenv(
            "CAMB_WORD_LEVEL_TIMESTAMPS", "false"
        ).lower() in {"1", "true", "yes", "on"}
        self._run_ids: Dict[str, int] = {}

    def _headers(self) -> Dict[str, str]:
        return {"x-api-key": self.api_key or ""}

    def _upload(self, audio_file: str, model: str, language: str, **kwargs) -> str:
        data: Dict[str, Any] = {
            "language": self._normalize_language(language),
        }
        self._add_optional_metadata(data)

        with open(audio_file, "rb") as media_file:
            response = requests.post(
                f"{self.base_url}/transcribe",
                headers=self._headers(),
                data=data,
                files={"media_file": (os.path.basename(audio_file), media_file)},
                timeout=120,
            )

        if response.status_code not in (200, 201):
            raise RuntimeError(f"CAMB.AI transcription task failed: {response.text}")

        task = response.json()
        task_id = task.get("task_id")
        if not task_id:
            raise RuntimeError(
                f"CAMB.AI task response did not include task_id: {task}"
            )
        return str(task_id)

    def _poll(self, job_id: str) -> str:
        response = requests.get(
            f"{self.base_url}/transcribe/{job_id}",
            headers=self._headers(),
            timeout=60,
        )
        if response.status_code != 200:
            raise RuntimeError(f"CAMB.AI status check failed: {response.text}")

        status_payload = response.json()
        status = str(status_payload.get("status", "")).upper()
        if status == "SUCCESS":
            run_id = status_payload.get("run_id")
            if run_id is None:
                raise RuntimeError(
                    f"CAMB.AI status response did not include run_id: {status_payload}"
                )
            self._run_ids[job_id] = int(run_id)
            return "completed"
        if status == "PENDING":
            return "pending"
        if status in {"ERROR", "TIMEOUT", "PAYMENT_REQUIRED"}:
            return "failed"
        return "pending"

    def _fetch_result(self, job_id: str) -> TranscriptionResult:
        run_id = self._run_ids.get(job_id)
        if run_id is None:
            raise RuntimeError(f"CAMB.AI run_id missing for task {job_id}")

        response = requests.get(
            f"{self.base_url}/transcription-result/{run_id}",
            headers=self._headers(),
            params={"word_level_timestamps": self.word_level_timestamps},
            timeout=60,
        )
        if response.status_code != 200:
            raise RuntimeError(f"CAMB.AI result fetch failed: {response.text}")

        payload = response.json()
        segments = self._extract_segments(payload)
        text = self._extract_text(payload, segments)
        return TranscriptionResult(
            text=text,
            segments=segments or None,
            raw_response=payload,
        )

    def _add_optional_metadata(self, data: Dict[str, Any]) -> None:
        project_name = os.getenv("CAMB_PROJECT_NAME")
        project_description = os.getenv("CAMB_PROJECT_DESCRIPTION")
        folder_id = os.getenv("CAMB_FOLDER_ID")

        if project_name:
            data["project_name"] = project_name
        if project_description:
            data["project_description"] = project_description
        if folder_id:
            data["folder_id"] = folder_id

    def _normalize_language(self, language: Optional[str]) -> str:
        if not language:
            return "en-us"

        normalized = language.strip().lower().replace("_", "-")
        return LANGUAGE_ALIASES.get(normalized, normalized)

    def _extract_segments(self, payload: Any) -> List[Dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]

        if not isinstance(payload, dict):
            return []

        for key in ("transcript", "segments", "dialogue"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]

        result = payload.get("result")
        if isinstance(result, list):
            return [item for item in result if isinstance(item, dict)]
        if isinstance(result, dict):
            return self._extract_segments(result)

        return []

    def _extract_text(self, payload: Any, segments: List[Dict[str, Any]]) -> str:
        if segments:
            return "\n".join(
                str(segment.get("text", "")).strip()
                for segment in segments
                if segment.get("text")
            )

        if isinstance(payload, dict):
            text = payload.get("text")
            if isinstance(text, str):
                return text

        if isinstance(payload, list):
            return "\n".join(str(item).strip() for item in payload if item)

        return ""
