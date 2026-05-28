# ABOUTME: Vosk offline transcription provider
# ABOUTME: Uses vosk package for offline speech recognition with local models

import json
import os
import wave
from typing import Optional

from sapat.providers import register
from sapat.providers.base import (
    AudioFormat,
    ProviderConfig,
    TranscriptionProvider,
    TranscriptionResult,
)


@register
class VoskProvider(TranscriptionProvider):
    name = "vosk"
    config = ProviderConfig(
        required_env_vars=[],
        required_packages=["vosk"],
        extras_key="vosk",
        max_file_size_mb=float("inf"),
        preferred_format=AudioFormat.WAV,
        supports_correction=False,
        default_model="vosk-model-small-en-us-0.15",
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
            import vosk
        except ImportError as exc:
            raise ImportError(
                "Vosk support requires the optional vosk package. "
                "Install it with `pip install 'sapat[vosk]'` or `pip install vosk`."
            ) from exc

        model_path = kwargs.get("model_path", os.getenv("VOSK_MODEL_PATH", model))
        sample_rate = int(os.getenv("VOSK_SAMPLE_RATE", "16000"))
        chunk_size = int(os.getenv("VOSK_CHUNK_SIZE", "4000"))

        self._validate_config(audio_file, model_path)

        vosk_model = vosk.Model(model_path)
        transcript_parts = []

        with wave.open(audio_file, "rb") as audio:
            recognizer = vosk.KaldiRecognizer(vosk_model, audio.getframerate())

            while True:
                data = audio.readframes(chunk_size)
                if len(data) == 0:
                    break
                if recognizer.AcceptWaveform(data):
                    transcript_parts.append(
                        self._extract_text(recognizer.Result())
                    )

            transcript_parts.append(
                self._extract_text(recognizer.FinalResult())
            )

        text = " ".join(part for part in transcript_parts if part).strip()
        return TranscriptionResult(text=text)

    def _validate_config(self, audio_file: str, model_path: str):
        if not os.path.exists(audio_file):
            raise ValueError(f"File {audio_file} does not exist.")
        if not model_path:
            raise ValueError(
                "VOSK_MODEL_PATH must point to an unpacked Vosk model directory."
            )
        if not os.path.isdir(model_path):
            raise ValueError(
                f"VOSK_MODEL_PATH does not exist or is not a directory: {model_path}"
            )

    @staticmethod
    def _extract_text(result_json: str):
        try:
            return json.loads(result_json).get("text", "")
        except json.JSONDecodeError:
            return ""
