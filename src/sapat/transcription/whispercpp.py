import os
import subprocess
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from .base import TranscriptionBase


load_dotenv(".env")


class WhisperCppTranscription(TranscriptionBase):
    """
    Local whisper.cpp CLI implementation for transcription.
    """

    def __init__(self, temperature: float):
        self.binary = os.getenv("WHISPER_CPP_BINARY", "whisper-cli")
        self.model_path = os.getenv("WHISPER_CPP_MODEL_PATH")
        self.threads = os.getenv("WHISPER_CPP_THREADS")
        self.temperature = temperature

    def transcribe_audio(self, audio_file: str, **kwargs):
        self._validate_configuration(audio_file)

        with tempfile.TemporaryDirectory(prefix="sapat-whispercpp-") as tmpdir:
            tmpdir_path = Path(tmpdir)
            wav_file = tmpdir_path / f"{Path(audio_file).stem}.wav"
            output_base = tmpdir_path / "transcript"

            self._convert_mp3_to_wav(audio_file, wav_file)
            result = self._run_whispercpp(wav_file, output_base, **kwargs)

            output_txt = output_base.with_suffix(".txt")
            if output_txt.exists():
                return output_txt.read_text(encoding="utf-8").strip()

            stdout = (result.stdout or "").strip()
            if stdout:
                return stdout

            raise Exception("whisper.cpp completed without producing transcript text.")

    def _convert_mp3_to_wav(self, audio_file: str, wav_file: Path):
        command = [
            "ffmpeg",
            "-y",
            "-i",
            audio_file,
            "-ar",
            "16000",
            "-ac",
            "1",
            str(wav_file),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)

    def _run_whispercpp(self, wav_file: Path, output_base: Path, **kwargs):
        command = [
            self.binary,
            "-m",
            self.model_path,
            "-f",
            str(wav_file),
            "-otxt",
            "-of",
            str(output_base),
            "-nt",
        ]

        language = kwargs.get("language")
        if language:
            command.extend(["-l", language])

        prompt = kwargs.get("prompt")
        if prompt:
            command.extend(["--prompt", prompt])

        if self.threads:
            command.extend(["-t", self.threads])

        try:
            return subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"whisper.cpp binary '{self.binary}' was not found. "
                "Set WHISPER_CPP_BINARY to your whisper-cli path."
            ) from exc

    def _validate_configuration(self, audio_file: str):
        if not Path(audio_file).exists():
            raise ValueError(f"File {audio_file} does not exist.")

        if not self.model_path:
            raise ValueError(
                "WHISPER_CPP_MODEL_PATH must point to a whisper.cpp model file."
            )

        if not Path(self.model_path).exists():
            raise ValueError(f"whisper.cpp model file does not exist: {self.model_path}")

    @staticmethod
    def generate_corrected_transcript(audio_file, temperature, prompt):
        raise NotImplementedError("Correction is not implemented for whisper.cpp.")
