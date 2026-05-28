# ABOUTME: Picovoice Leopard on-device transcription provider
# ABOUTME: Uses pvleopard package for local speech recognition

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
class PicovoiceProvider(TranscriptionProvider):
    name = "picovoice"
    config = ProviderConfig(
        required_env_vars=["PICOVOICE_ACCESS_KEY"],
        required_packages=["pvleopard"],
        extras_key="picovoice",
        max_file_size_mb=float("inf"),
        preferred_format=AudioFormat.WAV,
        supports_correction=False,
        default_model="leopard",
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
        access_key = os.getenv("PICOVOICE_ACCESS_KEY")
        if not access_key:
            raise ValueError("PICOVOICE_ACCESS_KEY is required for the picovoice provider.")

        self._validate_audio_file(audio_file)

        factory = self._get_leopard_factory()

        model_path = os.getenv("PICOVOICE_LEOPARD_MODEL_PATH") or None
        device = os.getenv("PICOVOICE_LEOPARD_DEVICE") or None
        enable_automatic_punctuation = self._bool_from_env(
            "PICOVOICE_LEOPARD_ENABLE_PUNCTUATION", True
        )
        enable_diarization = self._bool_from_env(
            "PICOVOICE_LEOPARD_ENABLE_DIARIZATION", False
        )

        leopard = factory(
            access_key=access_key,
            model_path=kwargs.get("model_path", model_path),
            device=kwargs.get("device", device),
            enable_automatic_punctuation=kwargs.get(
                "enable_automatic_punctuation", enable_automatic_punctuation
            ),
            enable_diarization=kwargs.get(
                "enable_diarization", enable_diarization
            ),
        )

        try:
            transcript, _words = leopard.process_file(audio_file)
        finally:
            delete = getattr(leopard, "delete", None)
            if callable(delete):
                delete()

        return TranscriptionResult(text=transcript)

    def _get_leopard_factory(self):
        try:
            import pvleopard
        except ImportError as exc:
            raise ImportError(
                "Picovoice Leopard support requires pvleopard. "
                "Install it with `pip install 'sapat[picovoice]'`."
            ) from exc

        return pvleopard.create

    @staticmethod
    def _validate_audio_file(audio_file: str):
        if not os.path.exists(audio_file):
            raise ValueError(f"File {audio_file} does not exist.")

    @staticmethod
    def _bool_from_env(name: str, default: bool = False) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}
