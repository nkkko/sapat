# ABOUTME: Mistral transcription provider with correction support
# ABOUTME: Uses Mistral's Voxtral endpoint for STT and chat endpoint for correction

import os
from typing import Optional

import requests

from sapat.providers import register
from sapat.providers.base import (
    AudioFormat,
    ProviderConfig,
    TranscriptionProvider,
    TranscriptionResult,
)


@register
class MistralProvider(TranscriptionProvider):
    name = "mistral"
    config = ProviderConfig(
        required_env_vars=["MISTRAL_API_KEY"],
        default_model="voxtral-mini-latest",
        supports_correction=True,
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
        url = "https://api.mistral.ai/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {os.getenv('MISTRAL_API_KEY')}"}

        data = {
            "model": model,
            "temperature": temperature,
            "language": language,
        }
        if prompt:
            data["context_bias"] = prompt

        with open(audio_file, "rb") as f:
            response = requests.post(
                url,
                headers=headers,
                data=data,
                files={"file": f},
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"Mistral transcription failed ({response.status_code}): {response.text}"
            )

        result = response.json()
        return TranscriptionResult(text=result.get("text", ""))

    def correct_transcript(self, text: str, temperature: float = 0) -> str:
        url = "https://api.mistral.ai/v1/chat/completions"
        api_key = os.getenv("MISTRAL_API_KEY")
        chat_model = os.getenv("MISTRAL_CHAT_MODEL", "mistral-small-latest")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        system_prompt = (
            "You are a helpful assistant. Your task is to correct any "
            "spelling discrepancies in the transcribed text. Make sure "
            "that the names of the products and persons are spelled "
            "correctly. Only add necessary punctuation such as periods, "
            "commas, and capitalization, and use only the context provided."
        )

        payload = {
            "model": chat_model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
        }

        response = requests.post(url, headers=headers, json=payload)

        if response.status_code != 200:
            raise RuntimeError(
                f"Mistral correction failed ({response.status_code}): {response.text}"
            )

        result = response.json()
        return result["choices"][0]["message"]["content"]
