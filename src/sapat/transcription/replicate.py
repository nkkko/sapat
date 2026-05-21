import os
from dotenv import load_dotenv
import replicate
from .base import TranscriptionBase

# Load environment variables
load_dotenv(".env")


class ReplicateTranscription(TranscriptionBase):
    """
    Replicate API implementation for transcription.
    """

    def __init__(self, temperature: float):
        """
        Initializes the ReplicateTranscription class.

        Parameters:
        - temperature (float): Accepted for CLI compatibility. Replicate's
          default Whisper model does not expose temperature.
        """
        self.api_token = os.getenv("REPLICATE_API_TOKEN")
        self.model_ref = os.getenv("REPLICATE_MODEL", "openai/whisper")
        self.translate = os.getenv("REPLICATE_TRANSLATE", "false").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.max_file_size_mb = int(os.getenv("REPLICATE_MAX_FILE_SIZE_MB", "100"))
        self.temperature = temperature

    def transcribe_audio(self, audio_file: str, **kwargs):
        """
        Transcribes an audio file using a Replicate-hosted speech-to-text model.

        Parameters:
        - audio_file (str): Path to the audio file.
        - kwargs: Additional parameters for transcription.

        Returns:
        - dict: A Sapat-compatible transcription result with a `text` key.
        """
        self._validate_audio_file(audio_file)
        if not self.api_token:
            raise ValueError("REPLICATE_API_TOKEN must be set to use the Replicate API.")

        client = replicate.Client(api_token=self.api_token)

        with open(audio_file, "rb") as audio:
            output = client.run(
                self.model_ref,
                input=self._build_input(audio, kwargs),
            )

        return {"text": self._extract_transcription_text(output)}

    def _build_input(self, audio, kwargs):
        payload = {"audio": audio}
        whisper_model = os.getenv("REPLICATE_WHISPER_MODEL")
        if whisper_model:
            payload["model"] = whisper_model

        translate = kwargs.get("translate", self.translate)
        if translate:
            payload["translate"] = bool(translate)

        return payload

    def _extract_transcription_text(self, output):
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

        raise ValueError(f"Unsupported Replicate transcription output: {output!r}")

    def _validate_audio_file(self, audio_file: str):
        if not os.path.exists(audio_file):
            raise ValueError(f"File {audio_file} does not exist.")

        file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            raise ValueError(
                f"File size exceeds the Replicate upload limit of {self.max_file_size_mb} MB."
            )

        valid_extensions = [".mp3", ".wav", ".flac", ".m4a", ".ogg"]
        if not any(str(audio_file).lower().endswith(ext) for ext in valid_extensions):
            raise ValueError(
                f"Unsupported audio file format: {audio_file}. Supported formats are {valid_extensions}."
            )
