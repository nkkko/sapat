# ABOUTME: Baidu Speech Recognition transcription provider
# ABOUTME: Uses Baidu's short-form STT API with OAuth token-based auth

import base64
import os
from typing import Dict, Optional

import requests

from sapat.providers import register
from sapat.providers.base import (
    AudioFormat,
    ProviderConfig,
    TranscriptionProvider,
    TranscriptionResult,
)

TOKEN_ENDPOINT = "https://aip.baidubce.com/oauth/2.0/token"
TRANSCRIPTION_ENDPOINT = "https://vop.baidu.com/server_api"

LANGUAGE_DEV_PIDS: Dict[str, int] = {
    "zh": 1537,
    "zh-cn": 1537,
    "zh_cn": 1537,
    "cmn": 1537,
    "en": 1737,
    "en-us": 1737,
    "en_us": 1737,
}


@register
class BaiduProvider(TranscriptionProvider):
    name = "baidu"
    config = ProviderConfig(
        required_env_vars=["BAIDU_API_KEY", "BAIDU_SECRET_KEY"],
        max_file_size_mb=10.0,
        preferred_format=AudioFormat.WAV,
        supports_correction=False,
        default_model="short_form",
    )

    def transcribe(
        self,
        audio_file: str,
        model: str,
        language: str = "en",
        prompt: Optional[str] = None,
        temperature: float = 0,
        **kwargs,
    ) -> TranscriptionResult:
        api_key = os.getenv("BAIDU_API_KEY", "")
        secret_key = os.getenv("BAIDU_SECRET_KEY", "")
        token = self._get_access_token(api_key, secret_key)

        with open(audio_file, "rb") as f:
            audio_data = f.read()

        audio_format = kwargs.get("format", "wav")
        sample_rate = int(kwargs.get("rate", 16000))

        payload = {
            "format": audio_format,
            "rate": sample_rate,
            "channel": 1,
            "token": token,
            "len": len(audio_data),
            "speech": base64.b64encode(audio_data).decode("utf-8"),
            "dev_pid": self._resolve_dev_pid(language),
        }

        response = requests.post(
            TRANSCRIPTION_ENDPOINT,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        result = response.json()

        if response.status_code == 200 and result.get("err_no") == 0:
            transcript = result.get("result", "")
            if isinstance(transcript, list):
                transcript = " ".join(transcript)
            return TranscriptionResult(text=transcript)

        error_message = result.get("err_msg", response.text)
        raise RuntimeError(
            f"Baidu transcription failed: {error_message}"
        )

    def resolve_model(self, model_alias: str) -> str:
        # Baidu uses dev_pid rather than model names
        return model_alias

    def _get_access_token(self, api_key: str, secret_key: str) -> str:
        response = requests.get(
            TOKEN_ENDPOINT,
            params={
                "grant_type": "client_credentials",
                "client_id": api_key,
                "client_secret": secret_key,
            },
            timeout=30,
        )
        result = response.json()

        if response.status_code == 200 and result and result.get("access_token"):
            return result["access_token"]

        error_message = (result or {}).get("error_description", response.text)
        raise RuntimeError(
            f"Could not fetch Baidu access token: {error_message}"
        )

    def _resolve_dev_pid(self, language: Optional[str]) -> int:
        if not language:
            return LANGUAGE_DEV_PIDS["en"]

        normalized = language.lower()
        return LANGUAGE_DEV_PIDS.get(
            normalized,
            LANGUAGE_DEV_PIDS.get(
                normalized.split("-")[0], LANGUAGE_DEV_PIDS["en"]
            ),
        )
