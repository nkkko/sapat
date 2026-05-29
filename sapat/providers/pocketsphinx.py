# ABOUTME: PocketSphinx offline transcription provider
# ABOUTME: Uses the pocketsphinx Python bindings for local WAV recognition

import os
import wave
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
class PocketSphinxProvider(TranscriptionProvider):
    name = "pocketsphinx"
    config = ProviderConfig(
        required_env_vars=[],
        required_packages=["pocketsphinx"],
        extras_key="pocketsphinx",
        max_file_size_mb=float("inf"),
        preferred_format=AudioFormat.WAV,
        supports_correction=False,
        default_model="default",
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
            from pocketsphinx import Pocketsphinx, Segmenter
        except ImportError as exc:
            raise ImportError(
                "PocketSphinx support requires the optional pocketsphinx package. "
                "Install it with `pip install 'sapat[pocketsphinx]'` or "
                "`pip install pocketsphinx`."
            ) from exc

        options = self._build_recognizer_options(audio_file, model, **kwargs)
        recognizer = Pocketsphinx(**options)

        transcript_parts = []
        with wave.open(audio_file, "rb") as audio:
            self._validate_wav(audio_file, audio)
            segmenter = Segmenter(sample_rate=audio.getframerate())
            for segment in segmenter.segment(audio.getfp()):
                recognizer.start_utt()
                recognizer.process_raw(segment.pcm, full_utt=True)
                recognizer.end_utt()

                if options.get("keyphrase") and recognizer.hyp() is None:
                    continue

                text = recognizer.hypothesis()
                if text:
                    transcript_parts.append(text)

        return TranscriptionResult(
            text=" ".join(transcript_parts).strip(),
            language=language,
        )

    def _build_recognizer_options(self, audio_file: str, model: str, **kwargs):
        path = Path(audio_file)
        if not path.is_file():
            raise ValueError(f"Audio file not found: {audio_file}")

        options = {}

        hmm = kwargs.get("hmm") or os.getenv("POCKETSPHINX_HMM")
        lm = kwargs.get("lm") or os.getenv("POCKETSPHINX_LM")
        dictionary = kwargs.get("dictionary") or os.getenv("POCKETSPHINX_DICT")
        keyphrase = kwargs.get("keyphrase") or os.getenv("POCKETSPHINX_KEYPHRASE")
        keywords = kwargs.get("kws") or os.getenv("POCKETSPHINX_KWS")
        threshold = kwargs.get("kws_threshold") or os.getenv(
            "POCKETSPHINX_KWS_THRESHOLD"
        )

        keyword_search = bool(keyphrase or keywords)
        if keyword_search:
            options["lm"] = False

        if (
            model
            and model != self.config.default_model
            and not lm
            and not keyword_search
        ):
            lm = model

        self._add_existing_path(options, "hmm", hmm)
        self._add_existing_path(options, "lm", lm)
        self._add_existing_path(options, "dict", dictionary)
        self._add_existing_path(options, "kws", keywords)

        if keyphrase:
            options["keyphrase"] = keyphrase
        if threshold:
            try:
                options["kws_threshold"] = float(threshold)
            except ValueError as exc:
                raise ValueError(
                    "POCKETSPHINX_KWS_THRESHOLD must be a number."
                ) from exc

        return options

    @staticmethod
    def _validate_wav(audio_file: str, audio):
        if audio.getnchannels() != 1:
            raise ValueError(
                f"PocketSphinx requires mono WAV input, got {audio.getnchannels()} "
                f"channels in {audio_file}."
            )
        if audio.getsampwidth() != 2:
            raise ValueError(
                f"PocketSphinx requires 16-bit PCM WAV input, got "
                f"{audio.getsampwidth() * 8}-bit samples in {audio_file}."
            )

    @staticmethod
    def _add_existing_path(options: dict, option_name: str, value: Optional[str]):
        if not value:
            return

        if not Path(value).exists():
            raise ValueError(f"PocketSphinx {option_name} path not found: {value}")

        options[option_name] = value
