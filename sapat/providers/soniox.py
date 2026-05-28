# ABOUTME: Soniox transcription provider
# ABOUTME: Uses Soniox SDK for async submit -> wait -> get_transcript transcription

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
class SonioxProvider(TranscriptionProvider):
    name = "soniox"
    config = ProviderConfig(
        required_env_vars=["SONIOX_API_KEY"],
        required_packages=["soniox"],
        extras_key="soniox",
        max_file_size_mb=25.0,
        preferred_format=AudioFormat.MP3,
        supports_correction=False,
        default_model="stt-async-v4",
    )

    def resolve_model(self, model_alias: str) -> str:
        aliases = {
            "v2": "stt-async-v2",
            "v3": "stt-async-v3",
            "v4": "stt-async-v4",
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
        from soniox import SonioxClient

        api_key = os.getenv("SONIOX_API_KEY", "")
        os.environ["SONIOX_API_KEY"] = api_key

        client = SonioxClient()
        self.client = client

        transcription = client.stt.transcribe(
            model=model,
            file=audio_file,
        )
        client.stt.wait(transcription.id)
        transcript = client.stt.get_transcript(transcription.id)

        # Clean up the remote job
        destroy = os.getenv(
            "SONIOX_DESTROY_AFTER_TRANSCRIPTION", "true"
        ).lower() not in {"0", "false", "no"}
        if destroy:
            client.stt.destroy(transcription.id)

        return TranscriptionResult(text=transcript.text)
