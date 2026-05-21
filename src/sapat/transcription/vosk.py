import json
import os
import subprocess
import tempfile
import wave
from pathlib import Path

from dotenv import load_dotenv

from .base import TranscriptionBase

load_dotenv(".env")


class VoskTranscription(TranscriptionBase):
    """
    Offline Vosk implementation for local transcription.
    """

    def __init__(self, temperature: float = 0.0):
        self.model_path = os.getenv("VOSK_MODEL_PATH")
        self.sample_rate = int(os.getenv("VOSK_SAMPLE_RATE", "16000"))
        self.chunk_size = int(os.getenv("VOSK_CHUNK_SIZE", "4000"))
        self.temperature = temperature

    def transcribe_audio(self, audio_file: str, **kwargs):
        self._validate_config(audio_file)

        wav_file = self._convert_to_wav(audio_file)
        try:
            try:
                import vosk
            except ImportError as exc:
                raise RuntimeError(
                    "Vosk support requires the optional dependency. "
                    "Install it with `pip install 'sapat[vosk]'` or `pip install vosk`."
                ) from exc

            model = vosk.Model(self.model_path)
            transcript_parts = []

            with wave.open(wav_file, "rb") as audio:
                recognizer = vosk.KaldiRecognizer(model, audio.getframerate())

                while True:
                    data = audio.readframes(self.chunk_size)
                    if len(data) == 0:
                        break
                    if recognizer.AcceptWaveform(data):
                        transcript_parts.append(self._extract_text(recognizer.Result()))

                transcript_parts.append(self._extract_text(recognizer.FinalResult()))

            return {"text": " ".join(part for part in transcript_parts if part).strip()}
        finally:
            wav_path = Path(wav_file)
            if wav_path.exists():
                wav_path.unlink()

    def _validate_config(self, audio_file: str):
        if not os.path.exists(audio_file):
            raise ValueError(f"File {audio_file} does not exist.")
        if not self.model_path:
            raise ValueError("VOSK_MODEL_PATH must point to an unpacked Vosk model directory.")
        if not os.path.isdir(self.model_path):
            raise ValueError(f"VOSK_MODEL_PATH does not exist or is not a directory: {self.model_path}")

    def _convert_to_wav(self, audio_file: str):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name

        command = [
            "ffmpeg",
            "-y",
            "-i",
            audio_file,
            "-ac",
            "1",
            "-ar",
            str(self.sample_rate),
            "-f",
            "wav",
            wav_path,
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return wav_path

    @staticmethod
    def _extract_text(result_json: str):
        try:
            return json.loads(result_json).get("text", "")
        except json.JSONDecodeError:
            return ""
