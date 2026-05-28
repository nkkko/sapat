# ABOUTME: Google Gemini transcription provider
# ABOUTME: Uses Gemini's generateContent REST API with inline base64 audio

import base64
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from sapat.providers import register
from sapat.providers.base import (
    AudioFormat,
    ProviderConfig,
    TranscriptionProvider,
    TranscriptionResult,
)

MIME_TYPES: Dict[str, str] = {
    ".mp3": "audio/mp3",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
}


@register
class GeminiProvider(TranscriptionProvider):
    name = "gemini"
    config = ProviderConfig(
        required_env_vars=["GOOGLE_API_KEY"],
        max_file_size_mb=14.0,
        preferred_format=AudioFormat.MP3,
        supports_correction=False,
        default_model="gemini-2.0-flash",
    )

    def resolve_model(self, model_alias: str) -> str:
        aliases = {
            "flash": "gemini-2.0-flash",
            "pro": "gemini-2.0-pro",
            "flash15": "gemini-1.5-flash",
        }
        return aliases.get(model_alias, model_alias)

    def transcribe(
        self,
        audio_file: str,
        model: str,
        language: str = "en",
        prompt: Optional[str] = None,
        temperature: float = 0,
        **kwargs,
    ) -> TranscriptionResult:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY", "")
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent"
        )
        audio_path = Path(audio_file)
        mime_type = MIME_TYPES[audio_path.suffix.lower()]
        inline_audio = self._encode_audio(audio_path)

        transcription_prompt = self._build_prompt(language=language, user_prompt=prompt)
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": transcription_prompt},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": inline_audio,
                            }
                        },
                    ],
                }
            ],
            "generationConfig": {
                "temperature": temperature,
            },
        }
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }

        timeout = kwargs.get("timeout", 120)
        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=timeout
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Gemini transcription request failed: {exc}"
            ) from exc

        response_payload = self._decode_response_json(response)
        if not 200 <= response.status_code < 300:
            error_message = self._extract_error_message(
                response_payload, response.text
            )
            raise RuntimeError(
                f"Gemini transcription failed with status "
                f"{response.status_code}: {error_message}"
            )

        text = self._extract_transcription_text(response_payload)
        return TranscriptionResult(text=text)

    def _build_prompt(
        self, language: Optional[str] = None, user_prompt: Optional[str] = None
    ) -> str:
        parts: List[str] = [
            "Transcribe the provided audio as accurately as possible.",
            "Return only the transcription text.",
        ]
        if language:
            parts.append(f"Language: {language}.")
        if user_prompt:
            parts.append(f"Additional transcription guidance: {user_prompt}")
        return "\n".join(parts)

    def _encode_audio(self, audio_path: Path) -> str:
        with open(audio_path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")

    def _decode_response_json(
        self, response: requests.Response
    ) -> Dict[str, Any]:
        try:
            return response.json()
        except ValueError as exc:
            if not 200 <= response.status_code < 300:
                raise RuntimeError(
                    f"Gemini transcription failed with status "
                    f"{response.status_code}: {response.text}"
                ) from exc
            raise RuntimeError(
                f"Gemini transcription returned a non-JSON response: "
                f"{response.text}"
            ) from exc

    def _extract_error_message(
        self, response_payload: Dict[str, Any], fallback: str
    ) -> str:
        error = response_payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if message:
                return str(message)
        if response_payload:
            return str(response_payload)
        return fallback or "unknown error"

    def _extract_transcription_text(
        self, response_payload: Dict[str, Any]
    ) -> str:
        candidates = response_payload.get("candidates") or []
        if not candidates:
            prompt_feedback = response_payload.get("promptFeedback")
            if prompt_feedback:
                raise RuntimeError(
                    f"Gemini transcription returned no candidates. "
                    f"Prompt feedback: {prompt_feedback}"
                )
            raise RuntimeError("Gemini transcription returned no candidates.")

        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        texts: List[str] = []
        for part in parts:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())

        transcription = "\n".join(texts).strip()
        if not transcription:
            finish_reason = candidates[0].get("finishReason")
            message = "Gemini transcription returned an empty response."
            if finish_reason:
                message += f" Finish reason: {finish_reason}."
            raise RuntimeError(message)

        return transcription
