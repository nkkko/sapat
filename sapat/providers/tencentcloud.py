# ABOUTME: Tencent Cloud ASR transcription provider
# ABOUTME: Uses signed SentenceRecognition requests for short local audio files

import base64
import datetime as dt
import hashlib
import hmac
import json
import os
import time
from typing import Dict, Optional

import requests

from sapat.providers import register
from sapat.providers.base import (
    AudioFormat,
    ProviderConfig,
    TranscriptionProvider,
    TranscriptionResult,
)

SERVICE = "asr"
VERSION = "2019-06-14"
ACTION = "SentenceRecognition"
DEFAULT_ENDPOINT = "asr.tencentcloudapi.com"
LANGUAGE_MODELS: Dict[str, str] = {
    "zh": "16k_zh",
    "zh-cn": "16k_zh",
    "zh_cn": "16k_zh",
    "cmn": "16k_zh",
    "en": "16k_en",
    "en-us": "16k_en",
    "en_us": "16k_en",
    "ja": "16k_ja",
    "ko": "16k_ko",
    "yue": "16k_yue",
    "vi": "16k_vi",
    "ms": "16k_ms",
    "id": "16k_id",
    "fil": "16k_fil",
    "th": "16k_th",
    "pt": "16k_pt",
    "tr": "16k_tr",
    "ar": "16k_ar",
    "es": "16k_es",
    "hi": "16k_hi",
    "fr": "16k_fr",
    "de": "16k_de",
}


@register
class TencentCloudProvider(TranscriptionProvider):
    name = "tencentcloud"
    config = ProviderConfig(
        required_env_vars=["TENCENTCLOUD_SECRET_ID", "TENCENTCLOUD_SECRET_KEY"],
        max_file_size_mb=3.0,
        preferred_format=AudioFormat.MP3,
        supports_correction=False,
        default_model="16k_zh",
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
        with open(audio_file, "rb") as f:
            audio_data = f.read()

        voice_format = kwargs.get("voice_format") or self._detect_voice_format(
            audio_file
        )
        payload = {
            "EngSerViceType": self.resolve_model(
                model or self._model_for_language(language)
            ),
            "SourceType": 1,
            "VoiceFormat": voice_format,
            "ProjectId": 0,
            "SubServiceType": 2,
            "UsrAudioKey": kwargs.get("usr_audio_key", "sapat-local-audio"),
            "Data": base64.b64encode(audio_data).decode("utf-8"),
            "DataLen": len(audio_data),
        }

        for key in (
            "WordInfo",
            "FilterDirty",
            "FilterModal",
            "FilterPunc",
            "ConvertNumMode",
            "HotwordId",
            "CustomizationId",
            "ReinforceHotword",
            "HotwordList",
            "InputSampleRate",
            "ReplaceTextId",
        ):
            value = kwargs.get(self._snake_case(key))
            if value is not None:
                payload[key] = value

        response = requests.post(
            self._endpoint_url(),
            headers=self._build_headers(payload),
            data=json.dumps(payload, separators=(",", ":")),
            timeout=60,
        )

        result = response.json()
        if response.status_code == 200 and "Response" in result:
            inner = result["Response"]
            if "Error" not in inner:
                return TranscriptionResult(
                    text=inner.get("Result", ""),
                    duration=self._duration_seconds(inner.get("AudioDuration")),
                    segments=inner.get("WordList"),
                    raw_response=result,
                )
            error = inner["Error"]
            message = error.get("Message") or error.get("Code") or response.text
        else:
            message = response.text

        raise RuntimeError(f"Tencent Cloud ASR transcription failed: {message}")

    def resolve_model(self, model_alias: str) -> str:
        aliases = {
            "default": "16k_zh",
            "zh": "16k_zh",
            "mandarin": "16k_zh",
            "en": "16k_en",
            "english": "16k_en",
            "cantonese": "16k_yue",
            "yue": "16k_yue",
            "multi": "16k_zh-PY",
        }
        return aliases.get((model_alias or "").lower(), model_alias)

    def _build_headers(self, payload: dict) -> dict:
        secret_id = os.getenv("TENCENTCLOUD_SECRET_ID", "")
        secret_key = os.getenv("TENCENTCLOUD_SECRET_KEY", "")
        region = os.getenv("TENCENTCLOUD_REGION", "ap-guangzhou")
        host = self._host()
        timestamp = int(os.getenv("TENCENTCLOUD_TIMESTAMP", str(int(time.time()))))
        date = dt.datetime.fromtimestamp(timestamp, dt.UTC).strftime("%Y-%m-%d")
        body = json.dumps(payload, separators=(",", ":"))
        hashed_payload = hashlib.sha256(body.encode("utf-8")).hexdigest()

        canonical_request = "\n".join(
            [
                "POST",
                "/",
                "",
                f"content-type:application/json; charset=utf-8\nhost:{host}\n",
                "content-type;host",
                hashed_payload,
            ]
        )
        credential_scope = f"{date}/{SERVICE}/tc3_request"
        string_to_sign = "\n".join(
            [
                "TC3-HMAC-SHA256",
                str(timestamp),
                credential_scope,
                hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
            ]
        )
        signature = self._sign(secret_key, date, string_to_sign)
        authorization = (
            "TC3-HMAC-SHA256 "
            f"Credential={secret_id}/{credential_scope}, "
            "SignedHeaders=content-type;host, "
            f"Signature={signature}"
        )

        return {
            "Authorization": authorization,
            "Content-Type": "application/json; charset=utf-8",
            "Host": host,
            "X-TC-Action": ACTION,
            "X-TC-Timestamp": str(timestamp),
            "X-TC-Version": VERSION,
            "X-TC-Region": region,
        }

    def _sign(self, secret_key: str, date: str, string_to_sign: str) -> str:
        secret_date = hmac.new(
            ("TC3" + secret_key).encode("utf-8"),
            date.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        secret_service = hmac.new(
            secret_date, SERVICE.encode("utf-8"), hashlib.sha256
        ).digest()
        secret_signing = hmac.new(
            secret_service, b"tc3_request", hashlib.sha256
        ).digest()
        return hmac.new(
            secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def _model_for_language(self, language: Optional[str]) -> str:
        if not language:
            return self.config.default_model
        normalized = language.strip().lower()
        return LANGUAGE_MODELS.get(
            normalized,
            LANGUAGE_MODELS.get(normalized.split("-")[0], self.config.default_model),
        )

    def _endpoint_url(self) -> str:
        return f"https://{self._host()}"

    def _host(self) -> str:
        return os.getenv("TENCENTCLOUD_ASR_ENDPOINT", DEFAULT_ENDPOINT)

    def _detect_voice_format(self, audio_file: str) -> str:
        suffix = os.path.splitext(audio_file)[1].lstrip(".").lower()
        return suffix or self.config.preferred_format.value

    def _duration_seconds(self, duration_ms):
        if duration_ms is None:
            return None
        return float(duration_ms) / 1000.0

    def _snake_case(self, value: str) -> str:
        chars = []
        for index, char in enumerate(value):
            if char.isupper() and index:
                chars.append("_")
            chars.append(char.lower())
        return "".join(chars)
