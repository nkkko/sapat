# ABOUTME: fal.ai transcription provider
# ABOUTME: Uses fal-client SDK to upload audio and run Whisper transcription

import os
from typing import Optional

from sapat.providers import register
from sapat.providers.base import (
    AudioFormat,
    ProviderConfig,
    TranscriptionProvider,
    TranscriptionResult,
)


@register
class FalAIProvider(TranscriptionProvider):
    name = "falai"
    config = ProviderConfig(
        required_env_vars=["FAL_KEY"],
        required_packages=["fal_client"],
        extras_key="falai",
        max_file_size_mb=25.0,
        preferred_format=AudioFormat.MP3,
        supports_correction=False,
        default_model="fal-ai/whisper",
    )

    def resolve_model(self, model_alias: str) -> str:
        aliases = {
            "whisper": "fal-ai/whisper",
            "large": "fal-ai/whisper/large",
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
        import fal_client

        api_key = os.getenv("FAL_KEY", "")
        os.environ["FAL_KEY"] = api_key

        audio_url = fal_client.upload_file(audio_file)

        arguments = {
            "audio_url": audio_url,
            "task": kwargs.get("task", "transcribe"),
            "chunk_level": kwargs.get("chunk_level", "segment"),
            "batch_size": kwargs.get("batch_size", 64),
        }

        if language:
            arguments["language"] = language
        if prompt:
            arguments["prompt"] = prompt

        result = fal_client.subscribe(model, arguments=arguments)

        text = result.get("text", "") if isinstance(result, dict) else str(result)
        return TranscriptionResult(
            text=text,
            raw_response=result,
        )
