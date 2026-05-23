import os
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

import click
from dotenv import load_dotenv

from .base import TranscriptionBase


load_dotenv(".env")


class WhisperCppTranscription(TranscriptionBase):
    """
    Local whisper.cpp CLI implementation for offline transcription.
    """

    def __init__(self):
        self.binary = os.getenv("WHISPERCPP_BINARY", "whisper-cli")
        self.model_path = os.getenv("WHISPERCPP_MODEL_PATH")
        self.threads = os.getenv("WHISPERCPP_THREADS")
        self.extra_args = shlex.split(os.getenv("WHISPERCPP_EXTRA_ARGS", ""))

    def process_file(self, input_file, language, prompt, temperature, quality, correct):
        if correct:
            raise ValueError(
                "Transcript correction is not available for whisper.cpp. "
                "Run without --correct, then review the transcript locally."
            )

        input_path = Path(input_file)
        txt_file = input_path.with_suffix(".txt")

        click.echo(f"Processing {input_file}")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_file = Path(tmp.name)

        try:
            self.convert_to_wav(str(input_path), str(wav_file), quality)
            click.echo("Conversion to WAV completed")

            transcription_result = self.transcribe_audio(
                str(wav_file),
                language=language,
                prompt=prompt,
            )
            click.echo("Transcription completed")

            with open(txt_file, "w", encoding="utf-8") as f:
                f.write(transcription_result)
            click.echo(f"Transcription saved to {txt_file}")
        finally:
            wav_file.unlink(missing_ok=True)

    @staticmethod
    def convert_to_wav(input_file: str, output_file: str, quality: str):
        if quality == "L":
            sample_rate = "16000"
        elif quality in ["M", "H"]:
            sample_rate = "16000"
        else:
            raise ValueError("Invalid quality option. Choose from 'L', 'M', 'H'.")

        command = [
            "ffmpeg",
            "-y",
            "-i",
            input_file,
            "-vn",
            "-ar",
            sample_rate,
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            output_file,
        ]
        subprocess.run(command, check=True)

    def transcribe_audio(self, audio_file: str, **kwargs):
        self._validate_configuration(audio_file)

        command = [
            self.binary,
            "-m",
            self.model_path,
            "-f",
            audio_file,
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

        command.extend(self.extra_args)

        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        return self._clean_output(result.stdout)

    def _validate_configuration(self, audio_file: str):
        if not self.model_path:
            raise ValueError(
                "WHISPERCPP_MODEL_PATH must point to a local ggml model file."
            )

        if not Path(self.model_path).is_file():
            raise ValueError(f"Whisper.cpp model not found: {self.model_path}")

        if not Path(audio_file).is_file():
            raise ValueError(f"Audio file not found: {audio_file}")

        binary_path = Path(self.binary)
        if not binary_path.is_file() and shutil.which(self.binary) is None:
            raise ValueError(
                "whisper.cpp binary not found. Set WHISPERCPP_BINARY to "
                "whisper-cli or the compiled whisper.cpp binary path."
            )

    @staticmethod
    def _clean_output(output: str) -> str:
        cleaned_lines = []
        timestamp_pattern = re.compile(
            r"^\[[0-9:.]+ --> [0-9:.]+\]\s*(?P<text>.*)$"
        )

        for line in output.splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            match = timestamp_pattern.match(stripped)
            if match:
                stripped = match.group("text").strip()

            if stripped.startswith(("whisper_", "main:", "system_info:")):
                continue

            cleaned_lines.append(stripped)

        return "\n".join(cleaned_lines).strip()
