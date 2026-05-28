# ABOUTME: Azure OpenAI transcription provider
# ABOUTME: Supports Whisper STT and GPT-based transcript correction

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
class AzureProvider(TranscriptionProvider):
    name = "azure"
    config = ProviderConfig(
        required_env_vars=[
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_STT_API_VERSION",
        ],
        max_file_size_mb=25.0,
        preferred_format=AudioFormat.MP3,
        supports_correction=True,
        default_model="whisper",
    )

    def resolve_model(self, model_alias: str) -> str:
        aliases = {
            "w": os.getenv("AZURE_OPENAI_STT_MODEL_NAME", "whisper"),
            "whisper": os.getenv("AZURE_OPENAI_STT_MODEL_NAME", "whisper"),
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
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_STT_API_VERSION")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")

        url = f"{endpoint}/openai/deployments/{model}/audio/translations?api-version={api_version}"
        headers = {"api-key": api_key}
        data = {"response_format": "json", "temperature": temperature}
        if language:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt

        with open(audio_file, "rb") as f:
            files = {"file": f}
            response = requests.post(url, headers=headers, data=data, files=files)

        if response.status_code != 200:
            raise RuntimeError(f"Azure transcription failed ({response.status_code}): {response.text}")

        result = response.json()
        return TranscriptionResult(text=result.get("text", ""))

    def correct_transcript(self, text: str, temperature: float = 0) -> str:
        from openai import AzureOpenAI

        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION_CHAT"),
        )

        system_prompt = (
            "You are a helpful assistant. Your task is to correct any "
            "spelling discrepancies in the transcribed text. Make sure "
            "that the names of the products and persons are spelled "
            "correctly. Only add necessary punctuation such as periods, "
            "commas, and capitalization, and use only the context provided."
        )

        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_CHAT"),
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
        )
        return response.choices[0].message.content
