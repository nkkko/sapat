import os

from dotenv import load_dotenv

from .base import TranscriptionBase

load_dotenv(".env")


def _bool_from_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class PicovoiceLeopardTranscription(TranscriptionBase):
    """
    Picovoice Leopard implementation for local, on-device transcription.
    """

    def __init__(self, temperature: float, leopard_factory=None):
        self.temperature = temperature
        self.access_key = os.getenv("PICOVOICE_ACCESS_KEY")
        self.model_path = os.getenv("PICOVOICE_LEOPARD_MODEL_PATH") or None
        self.device = os.getenv("PICOVOICE_LEOPARD_DEVICE") or None
        self.enable_automatic_punctuation = _bool_from_env(
            "PICOVOICE_LEOPARD_ENABLE_PUNCTUATION", True
        )
        self.enable_diarization = _bool_from_env(
            "PICOVOICE_LEOPARD_ENABLE_DIARIZATION"
        )
        self._leopard_factory = leopard_factory

    def transcribe_audio(self, audio_file: str, **kwargs):
        """
        Transcribes an audio file using Picovoice Leopard.

        Parameters:
        - audio_file (str): Path to the local audio file.
        - kwargs: Optional provider overrides for model_path, device,
          enable_automatic_punctuation, and enable_diarization.

        Returns:
        - str: The transcript text.
        """
        self._validate_audio_file(audio_file)
        factory = self._get_leopard_factory()

        leopard = factory(
            access_key=self.access_key,
            model_path=kwargs.get("model_path", self.model_path),
            device=kwargs.get("device", self.device),
            enable_automatic_punctuation=kwargs.get(
                "enable_automatic_punctuation", self.enable_automatic_punctuation
            ),
            enable_diarization=kwargs.get(
                "enable_diarization", self.enable_diarization
            ),
        )

        try:
            transcript, _words = leopard.process_file(audio_file)
            return transcript
        finally:
            delete = getattr(leopard, "delete", None)
            if callable(delete):
                delete()

    def _get_leopard_factory(self):
        if not self.access_key:
            raise ValueError("PICOVOICE_ACCESS_KEY is required for --api leopard.")

        if self._leopard_factory is not None:
            return self._leopard_factory

        try:
            import pvleopard
        except ImportError as exc:
            raise RuntimeError(
                "Picovoice Leopard support requires pvleopard. "
                "Install it with `pip install 'sapat[leopard]'`."
            ) from exc

        return pvleopard.create

    @staticmethod
    def _validate_audio_file(audio_file: str):
        if not os.path.exists(audio_file):
            raise ValueError(f"File {audio_file} does not exist.")
