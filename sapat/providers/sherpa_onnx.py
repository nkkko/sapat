# ABOUTME: Local sherpa-onnx transcription provider
# ABOUTME: Supports explicit SenseVoice, Whisper, and transducer model bundles

import os
import wave
from pathlib import Path
from typing import Dict, Optional, Tuple

from sapat.providers import register
from sapat.providers.base import (
    AudioFormat,
    ProviderConfig,
    TranscriptionProvider,
    TranscriptionResult,
)


@register
class SherpaOnnxProvider(TranscriptionProvider):
    """Run offline ASR with a user-supplied sherpa-onnx model bundle.

    Model files are deliberately never downloaded by the provider. This keeps
    CLI runs reproducible and prevents a typo from triggering a large network
    transfer. Paths may be supplied as keyword arguments or environment
    variables, which also makes the provider straightforward to test.
    """

    name = "sherpa_onnx"
    config = ProviderConfig(
        required_env_vars=[],
        required_packages=["sherpa_onnx"],
        extras_key="sherpa_onnx",
        max_file_size_mb=float("inf"),
        preferred_format=AudioFormat.WAV,
        supports_correction=False,
        default_model="sense_voice",
    )
    _MODEL_FAMILIES = {"sense_voice", "whisper", "transducer"}

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
            import numpy as np
            import sherpa_onnx
        except ImportError as exc:
            raise ImportError(
                "sherpa-onnx support requires the optional sherpa-onnx package. "
                "Install it with `pip install 'sapat[sherpa_onnx]'`."
            ) from exc

        family = self._model_family(model, kwargs)
        paths = self._model_paths(family, kwargs)
        self._validate_files(audio_file, paths)

        threads = self._positive_int(
            kwargs.get("num_threads", os.getenv("SHERPA_ONNX_NUM_THREADS", "2")),
            "SHERPA_ONNX_NUM_THREADS",
        )
        provider = kwargs.get("execution_provider") or os.getenv(
            "SHERPA_ONNX_EXECUTION_PROVIDER", "cpu"
        )
        recognizer = self._create_recognizer(
            sherpa_onnx,
            family,
            paths,
            language=language,
            num_threads=threads,
            provider=provider,
        )

        samples, sample_rate, duration = self._read_wav(audio_file, np)
        stream = recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        recognizer.decode_stream(stream)

        result = stream.result
        text = getattr(result, "text", str(result)).strip()
        return TranscriptionResult(
            text=text,
            language=language or None,
            duration=duration,
            raw_response=result,
        )

    @classmethod
    def _model_family(cls, model: str, kwargs: Dict[str, object]) -> str:
        family = str(
            kwargs.get("model_type")
            or os.getenv("SHERPA_ONNX_MODEL_TYPE")
            or model
            or cls.config.default_model
        ).lower()
        if family not in cls._MODEL_FAMILIES:
            choices = ", ".join(sorted(cls._MODEL_FAMILIES))
            raise ValueError(
                f"Unsupported sherpa-onnx model family '{family}'. Choose: {choices}."
            )
        return family

    @staticmethod
    def _first(kwargs: Dict[str, object], key: str, env_var: str) -> str:
        value = kwargs.get(key) or os.getenv(env_var, "")
        return str(value).strip()

    @classmethod
    def _model_paths(cls, family: str, kwargs: Dict[str, object]) -> Dict[str, str]:
        paths = {
            "tokens": cls._first(kwargs, "tokens", "SHERPA_ONNX_TOKENS_PATH"),
        }
        if family == "sense_voice":
            paths["model"] = cls._first(kwargs, "model_path", "SHERPA_ONNX_MODEL_PATH")
        else:
            paths["encoder"] = cls._first(kwargs, "encoder", "SHERPA_ONNX_ENCODER_PATH")
            paths["decoder"] = cls._first(kwargs, "decoder", "SHERPA_ONNX_DECODER_PATH")
            if family == "transducer":
                paths["joiner"] = cls._first(
                    kwargs, "joiner", "SHERPA_ONNX_JOINER_PATH"
                )
        return paths

    @staticmethod
    def _validate_files(audio_file: str, paths: Dict[str, str]) -> None:
        audio = Path(audio_file)
        if not audio.is_file():
            raise ValueError(f"Audio file does not exist: {audio_file}")
        if audio.suffix.lower() != ".wav":
            raise ValueError(
                "sherpa-onnx requires WAV input; let Sapat convert it first."
            )

        missing_settings = [name for name, path in paths.items() if not path]
        if missing_settings:
            joined = ", ".join(sorted(missing_settings))
            raise ValueError(f"Missing sherpa-onnx model path setting(s): {joined}.")

        missing_files = [path for path in paths.values() if not Path(path).is_file()]
        if missing_files:
            raise ValueError(
                "sherpa-onnx model file does not exist: " + ", ".join(missing_files)
            )

    @staticmethod
    def _positive_int(value: object, name: str) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a positive integer.") from exc
        if parsed < 1:
            raise ValueError(f"{name} must be a positive integer.")
        return parsed

    @staticmethod
    def _truthy(value: Optional[str]) -> bool:
        return str(value or "").strip().lower() in {"1", "true", "yes", "on"}

    @classmethod
    def _create_recognizer(
        cls,
        sherpa_onnx,
        family: str,
        paths: Dict[str, str],
        *,
        language: str,
        num_threads: int,
        provider: str,
    ):
        if family == "sense_voice":
            return sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=paths["model"],
                tokens=paths["tokens"],
                num_threads=num_threads,
                language=language or "auto",
                use_itn=cls._truthy(os.getenv("SHERPA_ONNX_USE_ITN", "true")),
                provider=provider,
            )
        if family == "whisper":
            return sherpa_onnx.OfflineRecognizer.from_whisper(
                encoder=paths["encoder"],
                decoder=paths["decoder"],
                tokens=paths["tokens"],
                language=language,
                task="transcribe",
                num_threads=num_threads,
                provider=provider,
            )
        return sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=paths["encoder"],
            decoder=paths["decoder"],
            joiner=paths["joiner"],
            tokens=paths["tokens"],
            num_threads=num_threads,
            decoding_method=os.getenv("SHERPA_ONNX_DECODING_METHOD", "greedy_search"),
            provider=provider,
        )

    @staticmethod
    def _read_wav(audio_file: str, np) -> Tuple[object, int, float]:
        try:
            with wave.open(audio_file, "rb") as wav:
                if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                    raise ValueError(
                        "sherpa-onnx expects mono 16-bit PCM WAV input; "
                        "let Sapat convert the source first."
                    )
                sample_rate = wav.getframerate()
                frame_count = wav.getnframes()
                samples = np.frombuffer(wav.readframes(frame_count), dtype=np.int16)
        except wave.Error as exc:
            raise ValueError(f"Invalid WAV input: {exc}") from exc

        normalized = samples.astype(np.float32) / 32768.0
        duration = frame_count / sample_rate if sample_rate else 0.0
        return normalized, sample_rate, duration
