import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from .base import TranscriptionBase


load_dotenv(".env")


class WhisperXTranscription(TranscriptionBase):
    """
    Local WhisperX CLI implementation for time-aligned transcription.
    """

    def __init__(self, temperature: float = 0):
        self.command = shlex.split(os.getenv("WHISPERX_BINARY", "whisperx"))
        self.model = os.getenv("WHISPERX_MODEL", "small")
        self.device = os.getenv("WHISPERX_DEVICE", "cpu")
        self.compute_type = os.getenv("WHISPERX_COMPUTE_TYPE", "int8")
        self.batch_size = os.getenv("WHISPERX_BATCH_SIZE", "4")
        self.align_model = os.getenv("WHISPERX_ALIGN_MODEL")
        self.hf_token = os.getenv("WHISPERX_HF_TOKEN")
        self.diarize = self._is_truthy(os.getenv("WHISPERX_DIARIZE"))
        self.min_speakers = os.getenv("WHISPERX_MIN_SPEAKERS")
        self.max_speakers = os.getenv("WHISPERX_MAX_SPEAKERS")
        self.extra_args = shlex.split(os.getenv("WHISPERX_EXTRA_ARGS", ""))
        self.temperature = temperature

    def transcribe_audio(self, audio_file: str, **kwargs):
        self._validate_audio_file(audio_file)

        with tempfile.TemporaryDirectory() as output_dir:
            command = self._build_command(audio_file, output_dir, kwargs)
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )

            if result.returncode != 0:
                details = result.stderr.strip() or result.stdout.strip()
                raise Exception(f"WhisperX transcription failed: {details}")

            return self._read_txt_output(output_dir, audio_file)

    def generate_corrected_transcript(self, audio_file, temperature, prompt):
        raise NotImplementedError(
            "Transcript correction is not implemented for WhisperX. "
            "Run without --correct, then review or correct the transcript separately."
        )

    def _build_command(self, audio_file: str, output_dir: str, kwargs: dict):
        temperature = kwargs.get("temperature", self.temperature)

        command = list(self.command) + [
            audio_file,
            "--model",
            self.model,
            "--device",
            self.device,
            "--compute_type",
            self.compute_type,
            "--batch_size",
            self.batch_size,
            "--output_dir",
            output_dir,
            "--output_format",
            "txt",
            "--verbose",
            "False",
            "--temperature",
            str(temperature),
        ]

        language = kwargs.get("language")
        if language:
            command.extend(["--language", language])

        prompt = kwargs.get("prompt")
        if prompt:
            command.extend(["--initial_prompt", prompt])

        if self.align_model:
            command.extend(["--align_model", self.align_model])

        if self.hf_token:
            command.extend(["--hf_token", self.hf_token])

        if self.diarize:
            command.append("--diarize")
            if self.min_speakers:
                command.extend(["--min_speakers", self.min_speakers])
            if self.max_speakers:
                command.extend(["--max_speakers", self.max_speakers])

        command.extend(self.extra_args)
        return command

    def _validate_audio_file(self, audio_file: str):
        if not self.command:
            raise ValueError("WHISPERX_BINARY cannot be empty.")

        executable = self.command[0]
        if not Path(executable).exists() and shutil.which(executable) is None:
            raise ValueError(
                "WhisperX executable was not found. Set WHISPERX_BINARY to "
                "the whisperx command or install it with `pip install whisperx`."
            )

        if not os.path.exists(audio_file):
            raise ValueError(f"File {audio_file} does not exist.")

    @staticmethod
    def _read_txt_output(output_dir: str, audio_file: str):
        expected = Path(output_dir) / (Path(audio_file).stem + ".txt")
        if expected.exists():
            return expected.read_text(encoding="utf-8").strip()

        txt_outputs = sorted(Path(output_dir).glob("*.txt"))
        if txt_outputs:
            return txt_outputs[0].read_text(encoding="utf-8").strip()

        raise Exception("WhisperX did not produce a .txt transcript.")

    @staticmethod
    def _is_truthy(value):
        return str(value).lower() in ("1", "true", "yes", "on")
