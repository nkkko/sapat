# ABOUTME: SpeechBrain local transcription provider
# ABOUTME: Uses the maintained SpeechBrain inference API for local ASR

import os
from pathlib import Path
from typing import Any, Optional

from sapat.providers import register
from sapat.providers.base import (
    AudioFormat,
    ProviderConfig,
    TranscriptionProvider,
    TranscriptionResult,
)


@register
class SpeechBrainProvider(TranscriptionProvider):
    """Transcribe audio locally with a pretrained SpeechBrain ASR model."""

    name = "speechbrain"
    config = ProviderConfig(
        required_env_vars=[],
        required_packages=["speechbrain"],
        extras_key="speechbrain",
        max_file_size_mb=float("inf"),
        preferred_format=AudioFormat.WAV,
        supports_correction=False,
        default_model="speechbrain/asr-crdnn-rnnlm-librispeech",
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
        self._validate_audio_file(audio_file)

        env_model = os.getenv("SPEECHBRAIN_MODEL")
        source = model or self.config.default_model
        if env_model and source == self.config.default_model:
            source = env_model
        savedir = os.getenv("SPEECHBRAIN_SAVEDIR")
        device = os.getenv("SPEECHBRAIN_DEVICE", "cpu")

        loader_kwargs = {
            "source": source,
            "run_opts": {"device": device},
        }
        if savedir:
            loader_kwargs["savedir"] = savedir

        try:
            asr_class = self._get_asr_class()
            recognizer = asr_class.from_hparams(**loader_kwargs)
            raw_output = recognizer.transcribe_file(audio_file)
            text = self._extract_text(raw_output)
        except ImportError:
            raise
        except Exception as exc:
            raise RuntimeError(f"SpeechBrain transcription failed: {exc}") from exc

        return TranscriptionResult(text=text, raw_response=raw_output)

    @staticmethod
    def _get_asr_class():
        try:
            from speechbrain.inference.ASR import EncoderDecoderASR
        except ImportError as exc:
            raise ImportError(
                "SpeechBrain support requires the optional speechbrain package. "
                "Install it with `pip install 'sapat[speechbrain]'`."
            ) from exc

        return EncoderDecoderASR

    @staticmethod
    def _validate_audio_file(audio_file: str) -> None:
        if not Path(audio_file).is_file():
            raise ValueError(f"File {audio_file} does not exist.")

    @classmethod
    def _extract_text(cls, output: Any) -> str:
        if isinstance(output, str):
            return output.strip()

        if isinstance(output, tuple):
            if not output:
                return ""
            return cls._extract_text(output[0])

        if isinstance(output, list):
            parts = [cls._extract_text(part) for part in output]
            return " ".join(part for part in parts if part).strip()

        if output is None:
            return ""

        raise ValueError(f"Unsupported SpeechBrain transcription output: {output!r}")
