# ABOUTME: Symbl.ai async audio transcription provider
# ABOUTME: Uses Symbl's async API: submit audio -> poll job -> fetch transcript messages

import mimetypes
import os
from typing import Dict, Optional

import requests

from sapat.providers import register
from sapat.providers.async_poll import AsyncPollProvider
from sapat.providers.base import (
    ProviderConfig,
    TranscriptionResult,
)

TOKEN_URL = "https://api.symbl.ai/oauth2/token:generate"

LANGUAGE_CODES = {
    "ar": "ar-SA",
    "de": "de-DE",
    "en": "en-US",
    "es": "es-ES",
    "fa": "fa-IR",
    "fr": "fr-FR",
    "hi": "hi-IN",
    "it": "it-IT",
    "ja": "ja-JP",
    "nl": "nl-NL",
    "pt": "pt-BR",
    "ru": "ru-RU",
}


@register
class SymblProvider(AsyncPollProvider):
    """Symbl.ai Async Audio API transcription provider."""

    name = "symbl"
    config = ProviderConfig(
        required_env_vars=[],
        default_model="default",
    )

    poll_interval: float = float(os.getenv("SYMBL_JOB_POLL_INTERVAL_SECONDS", "5"))
    max_poll_time: float = float(os.getenv("SYMBL_JOB_TIMEOUT_SECONDS", "600"))

    def __init__(self):
        super().__init__()
        self.base_url = (
            os.getenv("SYMBL_API_BASE_URL", "https://api.symbl.ai/v1")
        ).rstrip("/")
        self.access_token = os.getenv("SYMBL_ACCESS_TOKEN")
        self.app_id = os.getenv("SYMBL_APP_ID")
        self.app_secret = os.getenv("SYMBL_APP_SECRET")
        # conversation_id is stashed during _upload for use in _fetch_result
        self._conversation_id: Optional[str] = None

    @classmethod
    def is_available(cls) -> bool:
        has_token = bool(os.getenv("SYMBL_ACCESS_TOKEN"))
        has_credentials = bool(
            os.getenv("SYMBL_APP_ID") and os.getenv("SYMBL_APP_SECRET")
        )
        return has_token or has_credentials

    def _get_access_token(self) -> str:
        if self.access_token:
            return self.access_token

        if not self.app_id or not self.app_secret:
            raise ValueError(
                "Set SYMBL_ACCESS_TOKEN or both SYMBL_APP_ID and SYMBL_APP_SECRET."
            )

        response = requests.post(
            TOKEN_URL,
            json={
                "type": "application",
                "appId": self.app_id,
                "appSecret": self.app_secret,
            },
            timeout=30,
        )
        if response.status_code not in (200, 201):
            raise RuntimeError(f"Symbl token generation failed: {response.text}")

        token = response.json().get("accessToken")
        if not token:
            raise RuntimeError("Symbl token response did not include accessToken.")
        self.access_token = token
        return token

    def _headers(self, content_type: Optional[str] = None) -> Dict[str, str]:
        headers = {"Authorization": f"Bearer {self._get_access_token()}"}
        if content_type:
            headers["Content-Type"] = content_type
        return headers

    def _language_code(self, language: str) -> str:
        normalized = language.strip()
        if "-" in normalized:
            return normalized
        return LANGUAGE_CODES.get(normalized.lower(), normalized)

    def _content_type(self, audio_file: str) -> str:
        guessed, _ = mimetypes.guess_type(audio_file)
        if guessed:
            return guessed
        if audio_file.endswith(".mp3"):
            return "audio/mpeg"
        if audio_file.endswith(".wav"):
            return "audio/wav"
        return "application/octet-stream"

    def _upload(self, audio_file: str, model: str, language: str, **kwargs) -> str:
        with open(audio_file, "rb") as f:
            response = requests.post(
                f"{self.base_url}/process/audio",
                headers=self._headers(self._content_type(audio_file)),
                params={"languageCode": self._language_code(language)},
                data=f,
                timeout=120,
            )

        if response.status_code not in (200, 201):
            raise RuntimeError(f"Symbl audio processing failed: {response.text}")

        result = response.json()
        job_id = result.get("jobId")
        conversation_id = result.get("conversationId")
        if not job_id or not conversation_id:
            raise RuntimeError(
                f"Symbl process response missing jobId or conversationId: {result}"
            )
        self._conversation_id = conversation_id
        return job_id

    def _poll(self, job_id: str) -> str:
        response = requests.get(
            f"{self.base_url}/job/{job_id}",
            headers=self._headers(),
            timeout=30,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Symbl job status check failed: {response.text}")

        status = response.json().get("status")
        if status == "completed":
            return "completed"
        if status in {"failed", "error"}:
            return "failed"
        return "pending"

    def _fetch_result(self, job_id: str) -> TranscriptionResult:
        response = requests.get(
            f"{self.base_url}/conversations/{self._conversation_id}/messages",
            headers=self._headers(),
            timeout=30,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Symbl message retrieval failed: {response.text}")

        messages = response.json().get("messages", [])
        text = "\n".join(
            msg.get("text", "") for msg in messages if msg.get("text")
        )
        return TranscriptionResult(text=text)
