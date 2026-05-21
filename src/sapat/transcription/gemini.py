import base64
import os
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

from .base import TranscriptionBase


load_dotenv(".env")


class GeminiTranscription(TranscriptionBase):
    """
    Google Gemini Developer API implementation for transcription.
    """

    DEFAULT_MODEL = "gemini-2.0-flash"
    DEFAULT_ENDPOINT_TEMPLATE = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "{model}:generateContent"
    )
    MIME_TYPES = {
        ".mp3": "audio/mp3",
        ".wav": "audio/wav",
        ".flac": "audio/flac",
    }

    def __init__(self, temperature: float, response_format: str = "text"):
        """
        Initializes the GeminiTranscription class.

        Parameters:
        - temperature (float): Default temperature value for transcription.
        - response_format (str): Present for consistency with other providers.
        """
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is required to use the Gemini transcription API."
            )

        self.model = os.getenv("GEMINI_MODEL") or self.DEFAULT_MODEL
        self.endpoint_template = (
            os.getenv("GEMINI_API_ENDPOINT_TEMPLATE")
            or self.DEFAULT_ENDPOINT_TEMPLATE
        )
        self.temperature = temperature
        self.response_format = response_format
        # Gemini inline requests must stay under the request-size limit. Base64
        # expands audio by roughly 33%, so Sapat keeps raw files below 14 MB.
        self.max_raw_file_size_mb = 14
        self.timeout_seconds = 120

    def transcribe_audio(self, audio_file: str, **kwargs: Any) -> str:
        """
        Transcribes an audio file using Gemini generateContent with inline audio.

        Parameters:
        - audio_file (str): Path to the audio file.
        - kwargs: Optional language, prompt, model, temperature, and timeout.

        Returns:
        - str: The transcription text.
        """
        audio_path = self._validate_audio_file(audio_file)
        model = kwargs.get("model", self.model)
        url = self._build_endpoint(model)
        prompt = self._build_prompt(
            language=kwargs.get("language"), user_prompt=kwargs.get("prompt")
        )
        mime_type = self.MIME_TYPES[audio_path.suffix.lower()]
        inline_audio = self._encode_audio(audio_path)

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
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
                "temperature": kwargs.get("temperature", self.temperature),
            },
        }

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=kwargs.get("timeout", self.timeout_seconds),
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Gemini transcription request failed: {exc}"
            ) from exc

        response_payload = self._decode_response_json(response)
        if not 200 <= response.status_code < 300:
            error_message = self._extract_error_message(response_payload, response.text)
            raise RuntimeError(
                "Gemini transcription failed with status "
                f"{response.status_code}: {error_message}"
            )

        return self._extract_transcription_text(response_payload)

    def generate_corrected_transcript(
        self, audio_file: str, temperature: float, system_prompt: str
    ) -> str:
        """
        Re-transcribes audio with Gemini while asking for light transcript cleanup.
        """
        correction_prompt = (
            "First transcribe the audio faithfully. Then apply only light cleanup: "
            "fix obvious spelling, capitalization, and punctuation mistakes without "
            "adding new content.\n"
            f"Correction guidance: {system_prompt}"
        )
        return self.transcribe_audio(
            audio_file,
            temperature=temperature,
            prompt=correction_prompt,
        )

    def _validate_audio_file(self, audio_file: str) -> Path:
        audio_path = Path(audio_file)
        if not audio_path.exists():
            raise FileNotFoundError(f"File {audio_file} does not exist.")
        if not audio_path.is_file():
            raise ValueError(f"Path {audio_file} is not a file.")

        valid_extensions = sorted(self.MIME_TYPES.keys())
        if audio_path.suffix.lower() not in self.MIME_TYPES:
            raise ValueError(
                "Unsupported audio file format: "
                f"{audio_file}. Supported formats are {valid_extensions}."
            )

        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_raw_file_size_mb:
            raise ValueError(
                "File size exceeds Sapat's conservative Gemini inline audio "
                f"limit of {self.max_raw_file_size_mb} MB. Split the audio "
                "or use a provider flow that supports uploaded files."
            )

        return audio_path

    def _build_endpoint(self, model: str) -> str:
        try:
            return self.endpoint_template.format(model=model)
        except KeyError as exc:
            raise ValueError(
                "GEMINI_API_ENDPOINT_TEMPLATE may only use the {model} placeholder."
            ) from exc

    def _build_prompt(
        self, language: Optional[str] = None, user_prompt: Optional[str] = None
    ) -> str:
        prompt_parts = [
            "Transcribe the provided audio as accurately as possible.",
            "Return only the transcription text.",
        ]
        if language:
            prompt_parts.append(f"Language: {language}.")
        if user_prompt:
            prompt_parts.append(f"Additional transcription guidance: {user_prompt}")
        return "\n".join(prompt_parts)

    def _encode_audio(self, audio_path: Path) -> str:
        with open(audio_path, "rb") as audio:
            return base64.b64encode(audio.read()).decode("ascii")

    def _decode_response_json(self, response: requests.Response) -> Dict[str, Any]:
        try:
            return response.json()
        except ValueError as exc:
            if not 200 <= response.status_code < 300:
                raise RuntimeError(
                    "Gemini transcription failed with status "
                    f"{response.status_code}: {response.text}"
                ) from exc
            raise RuntimeError(
                "Gemini transcription returned a non-JSON response: "
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

    def _extract_transcription_text(self, response_payload: Dict[str, Any]) -> str:
        candidates = response_payload.get("candidates") or []
        if not candidates:
            prompt_feedback = response_payload.get("promptFeedback")
            if prompt_feedback:
                raise RuntimeError(
                    "Gemini transcription returned no candidates. "
                    f"Prompt feedback: {prompt_feedback}"
                )
            raise RuntimeError("Gemini transcription returned no candidates.")

        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        texts = []
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
