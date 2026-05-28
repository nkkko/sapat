# ABOUTME: Moonshine on-device transcription provider
# ABOUTME: Uses moonshine-voice package for local inference

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
class MoonshineProvider(TranscriptionProvider):
    name = "moonshine"
    config = ProviderConfig(
        required_env_vars=[],
        required_packages=["moonshine"],
        extras_key="moonshine",
        max_file_size_mb=float("inf"),
        preferred_format=AudioFormat.WAV,
        supports_correction=False,
        default_model="moonshine/base",
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
        try:
            from moonshine_voice import Transcriber, get_model_for_language, load_wav_file
        except ImportError as exc:
            raise ImportError(
                "Moonshine support requires the optional moonshine-voice package. "
                "Install it with `pip install 'sapat[moonshine]'` or "
                "`pip install moonshine-voice`."
            ) from exc

        model_path, model_arch = get_model_for_language(language)
        audio_data, sample_rate = load_wav_file(audio_file)

        transcriber = Transcriber(
            model_path=model_path,
            model_arch=model_arch,
            update_interval=kwargs.get("update_interval", 0.5),
        )

        try:
            transcript = transcriber.transcribe_without_streaming(
                audio_data, sample_rate
            )
            text = self._transcript_to_text(transcript)
        finally:
            close = getattr(transcriber, "close", None)
            if callable(close):
                close()

        return TranscriptionResult(text=text)

    @staticmethod
    def _transcript_to_text(transcript):
        lines = getattr(transcript, "lines", None)
        if lines is None:
            return str(transcript)

        ordered_lines = sorted(lines, key=lambda line: getattr(line, "start_time", 0.0))
        return "\n".join(
            getattr(line, "text", str(line)).strip()
            for line in ordered_lines
            if getattr(line, "text", str(line)).strip()
        )
