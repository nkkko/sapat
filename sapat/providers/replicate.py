# ABOUTME: Replicate-hosted Whisper transcription provider
# ABOUTME: Uses the replicate Python SDK to run speech-to-text models

import os
from typing import Optional

from sapat.providers import register
from sapat.providers.base import (
    ProviderConfig,
    TranscriptionProvider,
    TranscriptionResult,
)


@register
class ReplicateProvider(TranscriptionProvider):
    """Replicate-hosted model transcription using the replicate SDK."""

    name = "replicate"
    config = ProviderConfig(
        required_env_vars=["REPLICATE_API_TOKEN"],
        required_packages=["replicate"],
        extras_key="replicate",
        default_model="openai/whisper",
    )

    def __init__(self):
        super().__init__()
        self.api_token = os.getenv("REPLICATE_API_TOKEN")
        self.model_ref = os.getenv("REPLICATE_MODEL", "openai/whisper")
        self.translate = os.getenv("REPLICATE_TRANSLATE", "false").lower() in {
            "1", "true", "yes", "on",
        }
        self.max_file_size_mb = int(
            os.getenv("REPLICATE_MAX_FILE_SIZE_MB", "100")
        )

    def resolve_model(self, model_alias: str) -> str:
        aliases = {
            "whisper": "openai/whisper",
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
        import replicate

        client = replicate.Client(api_token=self.api_token)

        with open(audio_file, "rb") as audio:
            output = client.run(
                model,
                input=self._build_input(audio, kwargs),
            )

        return TranscriptionResult(text=self._extract_transcription_text(output))

    def _build_input(self, audio, kwargs: dict) -> dict:
        payload = {"audio": audio}
        whisper_model = os.getenv("REPLICATE_WHISPER_MODEL")
        if whisper_model:
            payload["model"] = whisper_model

        translate = kwargs.get("translate", self.translate)
        if translate:
            payload["translate"] = bool(translate)

        return payload

    def _extract_transcription_text(self, output) -> str:
        if isinstance(output, str):
            return output

        if isinstance(output, dict):
            for key in ("transcription", "text", "output"):
                value = output.get(key)
                if isinstance(value, str):
                    return value
            if isinstance(output.get("segments"), list):
                text = " ".join(
                    segment.get("text", "").strip()
                    for segment in output["segments"]
                    if isinstance(segment, dict) and segment.get("text")
                ).strip()
                if text:
                    return text

        if isinstance(output, list):
            return "".join(str(part) for part in output)

        raise ValueError(
            f"Unsupported Replicate transcription output: {output!r}"
        )
