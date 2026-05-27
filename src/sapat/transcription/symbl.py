import mimetypes
import os
import time
from typing import Dict, Optional

import requests
from dotenv import load_dotenv

from .base import TranscriptionBase


load_dotenv(".env")


class SymblTranscription(TranscriptionBase):
    """
    Symbl.ai Async Audio API implementation for transcription.
    """

    DEFAULT_BASE_URL = "https://api.symbl.ai/v1"
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

    def __init__(
        self,
        temperature: float,
        base_url: Optional[str] = None,
        poll_interval_seconds: Optional[float] = None,
        timeout_seconds: Optional[float] = None,
    ):
        self.temperature = temperature
        self.base_url = (base_url or os.getenv("SYMBL_API_BASE_URL") or self.DEFAULT_BASE_URL).rstrip("/")
        self.access_token = os.getenv("SYMBL_ACCESS_TOKEN")
        self.app_id = os.getenv("SYMBL_APP_ID")
        self.app_secret = os.getenv("SYMBL_APP_SECRET")
        self.poll_interval_seconds = self._float_env(
            "SYMBL_JOB_POLL_INTERVAL_SECONDS",
            poll_interval_seconds,
            5.0,
        )
        self.timeout_seconds = self._float_env("SYMBL_JOB_TIMEOUT_SECONDS", timeout_seconds, 600.0)

    @staticmethod
    def _float_env(name: str, value: Optional[float], default: float) -> float:
        if value is not None:
            return float(value)
        env_value = os.getenv(name)
        return float(env_value) if env_value else default

    def _get_access_token(self) -> str:
        if self.access_token:
            return self.access_token

        if not self.app_id or not self.app_secret:
            raise ValueError(
                "Set SYMBL_ACCESS_TOKEN or both SYMBL_APP_ID and SYMBL_APP_SECRET "
                "before using --api symbl."
            )

        response = requests.post(
            self.TOKEN_URL,
            json={
                "type": "application",
                "appId": self.app_id,
                "appSecret": self.app_secret,
            },
            timeout=30,
        )
        if response.status_code not in (200, 201):
            raise Exception(f"Symbl token generation failed: {response.text}")

        token = response.json().get("accessToken")
        if not token:
            raise Exception("Symbl token generation response did not include accessToken.")
        self.access_token = token
        return token

    def _headers(self, content_type: Optional[str] = None) -> Dict[str, str]:
        headers = {"Authorization": f"Bearer {self._get_access_token()}"}
        if content_type:
            headers["Content-Type"] = content_type
        return headers

    def _language_code(self, language: Optional[str]) -> str:
        if not language:
            return "en-US"

        normalized = language.strip()
        if "-" in normalized:
            return normalized
        return self.LANGUAGE_CODES.get(normalized.lower(), normalized)

    def _content_type(self, audio_file: str) -> str:
        guessed, _ = mimetypes.guess_type(audio_file)
        if guessed:
            return guessed

        if audio_file.endswith(".mp3"):
            return "audio/mpeg"
        if audio_file.endswith(".wav"):
            return "audio/wav"
        return "application/octet-stream"

    def transcribe_audio(self, audio_file: str, **kwargs):
        """
        Transcribes an audio file with Symbl.ai's async audio workflow.

        Symbl returns a job ID and conversation ID. This method waits for the
        job to complete, then reads the conversation messages as transcript
        text so Sapat's existing file-writing path can stay unchanged.
        """
        if not os.path.exists(audio_file):
            raise ValueError(f"File {audio_file} does not exist.")

        process_result = self._submit_audio(audio_file, language=kwargs.get("language"))
        job_id = process_result.get("jobId")
        conversation_id = process_result.get("conversationId")
        if not job_id or not conversation_id:
            raise Exception(f"Symbl process response missing jobId or conversationId: {process_result}")

        self._wait_for_job(job_id)
        return {"text": self._get_transcript(conversation_id)}

    def _submit_audio(self, audio_file: str, language: Optional[str]) -> dict:
        with open(audio_file, "rb") as f:
            response = requests.post(
                f"{self.base_url}/process/audio",
                headers=self._headers(self._content_type(audio_file)),
                params={"languageCode": self._language_code(language)},
                data=f,
                timeout=120,
            )

        if response.status_code not in (200, 201):
            raise Exception(f"Symbl audio processing failed: {response.text}")
        return response.json()

    def _wait_for_job(self, job_id: str) -> dict:
        deadline = time.monotonic() + self.timeout_seconds
        last_status = None

        while time.monotonic() < deadline:
            response = requests.get(
                f"{self.base_url}/job/{job_id}",
                headers=self._headers(),
                timeout=30,
            )
            if response.status_code != 200:
                raise Exception(f"Symbl job status failed: {response.text}")

            payload = response.json()
            last_status = payload.get("status")
            if last_status == "completed":
                return payload
            if last_status in {"failed", "error"}:
                raise Exception(f"Symbl job failed: {payload}")

            time.sleep(self.poll_interval_seconds)

        raise TimeoutError(f"Timed out waiting for Symbl job {job_id}; last status: {last_status}")

    def _get_transcript(self, conversation_id: str) -> str:
        response = requests.get(
            f"{self.base_url}/conversations/{conversation_id}/messages",
            headers=self._headers(),
            timeout=30,
        )
        if response.status_code != 200:
            raise Exception(f"Symbl message retrieval failed: {response.text}")

        messages = response.json().get("messages", [])
        return "\n".join(message.get("text", "") for message in messages if message.get("text"))
