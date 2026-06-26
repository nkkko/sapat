# ABOUTME: Hugging Face Inference Providers ASR transcription provider
# ABOUTME: Sends audio to the router endpoint and parses text/chunk responses

import base64
import os
from typing import Optional
from urllib.parse import quote

import requests

from sapat.providers import register
from sapat.providers.base import (
    ProviderConfig,
    TranscriptionProvider,
    TranscriptionResult,
)


@register
class HuggingFaceProvider(TranscriptionProvider):
    """Hugging Face Inference Providers automatic speech recognition provider."""

    name = "huggingface"
    config = ProviderConfig(
        required_env_vars=["HF_TOKEN"],
        default_model="openai/whisper-large-v3",
    )

    def __init__(self):
        super().__init__()
        self.token = os.getenv("HF_TOKEN")
        self.provider = os.getenv("HUGGINGFACE_PROVIDER", "hf-inference")
        self.api_base = os.getenv(
            "HUGGINGFACE_API_BASE",
            "https://router.huggingface.co",
        ).rstrip("/")
        self.timeout = float(os.getenv("HUGGINGFACE_TIMEOUT", "120"))

    def _endpoint(self, model: str) -> str:
        provider = quote(self.provider.strip("/"), safe="")
        model_path = quote(model.strip("/"), safe="/")
        return f"{self.api_base}/{provider}/models/{model_path}"

    def _payload(
        self,
        audio_file: str,
        temperature: float,
        **kwargs,
    ) -> dict:
        with open(audio_file, "rb") as f:
            encoded_audio = base64.b64encode(f.read()).decode("ascii")

        payload = {"inputs": encoded_audio}
        parameters = {}
        generation_parameters = {}

        if temperature:
            generation_parameters["temperature"] = temperature
        if generation_parameters:
            parameters["generation_parameters"] = generation_parameters

        return_timestamps = kwargs.get("return_timestamps")
        if return_timestamps is None:
            return_timestamps = os.getenv(
                "HUGGINGFACE_RETURN_TIMESTAMPS", ""
            ).lower() in {
                "1",
                "true",
                "yes",
            }
        if return_timestamps:
            parameters["return_timestamps"] = True

        if parameters:
            payload["parameters"] = parameters
        return payload

    @staticmethod
    def _extract_text(data) -> str:
        if isinstance(data, dict):
            if isinstance(data.get("text"), str):
                return data["text"]
            if isinstance(data.get("generated_text"), str):
                return data["generated_text"]
            if isinstance(data.get("error"), str):
                raise RuntimeError(
                    f"Hugging Face transcription failed: {data['error']}"
                )
        if isinstance(data, str):
            return data
        raise RuntimeError("Hugging Face response did not include transcript text.")

    def transcribe(
        self,
        audio_file: str,
        model: str,
        language: str = "en",
        prompt: Optional[str] = None,
        temperature: float = 0,
        **kwargs,
    ) -> TranscriptionResult:
        response = requests.post(
            self._endpoint(model),
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
            json=self._payload(audio_file, temperature, **kwargs),
            timeout=self.timeout,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Hugging Face transcription failed ({response.status_code}): {response.text}"
            )

        data = response.json()
        text = self._extract_text(data)
        segments = data.get("chunks") if isinstance(data, dict) else None
        return TranscriptionResult(text=text, segments=segments, raw_response=data)
