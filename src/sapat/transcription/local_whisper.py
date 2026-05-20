import os
from pathlib import Path

from dotenv import load_dotenv

from .base import TranscriptionBase

load_dotenv(".env")


class LocalWhisperTranscription(TranscriptionBase):
    """
    Local faster-whisper implementation for offline transcription.
    """

    def __init__(self, temperature: float, response_format: str = "json", model_factory=None):
        self.model_name = os.getenv("SAPAT_LOCAL_WHISPER_MODEL", "base")
        self.device = os.getenv("SAPAT_LOCAL_WHISPER_DEVICE", "cpu")
        self.compute_type = os.getenv("SAPAT_LOCAL_WHISPER_COMPUTE_TYPE", "int8")
        self.beam_size = int(os.getenv("SAPAT_LOCAL_WHISPER_BEAM_SIZE", "5"))
        self.temperature = temperature
        self.response_format = response_format
        self.model_factory = model_factory or self._load_model
        self._model = None

    def transcribe_audio(self, audio_file: str, **kwargs):
        self._validate_audio_file(audio_file)

        model = self._get_model()
        language = kwargs.get("language") or None
        prompt = kwargs.get("prompt") or None
        temperature = kwargs.get("temperature", self.temperature)

        segments, info = model.transcribe(
            audio_file,
            language=language,
            initial_prompt=prompt,
            temperature=temperature,
            beam_size=self.beam_size,
        )
        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())

        if kwargs.get("response_format", self.response_format) == "text":
            return text

        return {
            "text": text,
            "language": getattr(info, "language", language),
            "duration": getattr(info, "duration", None),
        }

    def _get_model(self):
        if self._model is None:
            self._model = self.model_factory()
        return self._model

    def _load_model(self):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "Local transcription requires faster-whisper. "
                "Install it with `pip install faster-whisper` before using `--api local`."
            ) from exc

        return WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
        )

    @staticmethod
    def _validate_audio_file(audio_file: str):
        path = Path(audio_file)
        if not path.exists():
            raise ValueError(f"File {audio_file} does not exist.")

        valid_extensions = [".mp3", ".wav", ".flac", ".m4a"]
        if path.suffix.lower() not in valid_extensions:
            raise ValueError(
                f"Unsupported audio file format: {audio_file}. "
                f"Supported formats are {valid_extensions}."
            )
