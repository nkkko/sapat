# ABOUTME: faster-whisper local transcription provider
# ABOUTME: Uses CTranslate2-backed Whisper inference without hosted APIs

import os
from pathlib import Path
from typing import Optional

from sapat.providers import register
from sapat.providers.base import (
    AudioFormat,
    ProviderConfig,
    TranscriptionProvider,
    TranscriptionResult,
)


@register
class FasterWhisperProvider(TranscriptionProvider):
    name = "faster_whisper"
    config = ProviderConfig(
        required_env_vars=[],
        required_packages=["faster_whisper"],
        extras_key="faster-whisper",
        max_file_size_mb=float("inf"),
        preferred_format=AudioFormat.WAV,
        supports_correction=False,
        default_model="small",
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
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ImportError(
                "faster-whisper support requires the optional faster-whisper "
                "package. Install it with `pip install 'sapat[faster-whisper]'` "
                "or `pip install faster-whisper`."
            ) from exc

        self._validate_audio_file(audio_file)

        model_instance = WhisperModel(
            model,
            device=os.getenv("FASTER_WHISPER_DEVICE", "auto"),
            compute_type=os.getenv("FASTER_WHISPER_COMPUTE_TYPE", "int8"),
            cpu_threads=self._int_from_env("FASTER_WHISPER_CPU_THREADS", 0),
            num_workers=self._int_from_env("FASTER_WHISPER_NUM_WORKERS", 1),
            download_root=os.getenv("FASTER_WHISPER_DOWNLOAD_ROOT") or None,
            local_files_only=self._bool_from_env("FASTER_WHISPER_LOCAL_FILES_ONLY"),
        )

        segments, info = model_instance.transcribe(
            audio_file,
            language=language or None,
            task=os.getenv("FASTER_WHISPER_TASK", "transcribe"),
            beam_size=self._int_from_env("FASTER_WHISPER_BEAM_SIZE", 5),
            temperature=temperature,
            initial_prompt=prompt,
            vad_filter=self._bool_from_env("FASTER_WHISPER_VAD_FILTER"),
            word_timestamps=self._bool_from_env("FASTER_WHISPER_WORD_TIMESTAMPS"),
            condition_on_previous_text=self._bool_from_env(
                "FASTER_WHISPER_CONDITION_ON_PREVIOUS_TEXT",
                default=True,
            ),
        )

        segment_records = self._collect_segments(segments)
        text = "\n".join(
            segment["text"] for segment in segment_records if segment["text"]
        ).strip()

        return TranscriptionResult(
            text=text,
            language=getattr(info, "language", language),
            duration=getattr(info, "duration", None),
            segments=segment_records,
            raw_response=info,
        )

    @staticmethod
    def _collect_segments(segments):
        records = []
        for segment in segments:
            text = getattr(segment, "text", "").strip()
            records.append(
                {
                    "start": getattr(segment, "start", None),
                    "end": getattr(segment, "end", None),
                    "text": text,
                }
            )
        return records

    @staticmethod
    def _validate_audio_file(audio_file: str):
        if not Path(audio_file).is_file():
            raise ValueError(f"Audio file not found: {audio_file}")

    @staticmethod
    def _bool_from_env(name: str, default: bool = False) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _int_from_env(name: str, default: int) -> int:
        value = os.getenv(name)
        if value is None or value == "":
            return default
        return int(value)
