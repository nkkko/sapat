# ABOUTME: WhisperX local transcription provider
# ABOUTME: Uses whisperx CLI subprocess for time-aligned speech recognition

import os
import shlex
import shutil
import subprocess
import tempfile
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
class WhisperXProvider(TranscriptionProvider):
    name = "whisperx"
    config = ProviderConfig(
        required_env_vars=[],
        required_packages=[],
        extras_key="whisperx",
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
        binary = self._split_command(os.getenv("WHISPERX_BINARY", "whisperx"))
        device = os.getenv("WHISPERX_DEVICE", "cpu")
        compute_type = os.getenv("WHISPERX_COMPUTE_TYPE", "int8")
        batch_size = os.getenv("WHISPERX_BATCH_SIZE", "4")
        align_model = os.getenv("WHISPERX_ALIGN_MODEL")
        hf_token = os.getenv("WHISPERX_HF_TOKEN")
        diarize = self._is_truthy(os.getenv("WHISPERX_DIARIZE"))
        min_speakers = os.getenv("WHISPERX_MIN_SPEAKERS")
        max_speakers = os.getenv("WHISPERX_MAX_SPEAKERS")
        extra_args = shlex.split(os.getenv("WHISPERX_EXTRA_ARGS", ""))

        self._validate(audio_file, binary)

        with tempfile.TemporaryDirectory() as output_dir:
            command = list(binary) + [
                audio_file,
                "--model", model,
                "--device", device,
                "--compute_type", compute_type,
                "--batch_size", batch_size,
                "--output_dir", output_dir,
                "--output_format", "txt",
                "--verbose", "False",
                "--temperature", str(temperature),
            ]

            if language:
                command.extend(["--language", language])
            if prompt:
                command.extend(["--initial_prompt", prompt])
            if align_model:
                command.extend(["--align_model", align_model])
            if hf_token:
                command.extend(["--hf_token", hf_token])
            if diarize:
                command.append("--diarize")
                if min_speakers:
                    command.extend(["--min_speakers", min_speakers])
                if max_speakers:
                    command.extend(["--max_speakers", max_speakers])

            command.extend(extra_args)

            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )

            if result.returncode != 0:
                details = result.stderr.strip() or result.stdout.strip()
                raise RuntimeError(
                    f"WhisperX transcription failed: {details}"
                )

            text = self._read_txt_output(output_dir, audio_file)
            return TranscriptionResult(text=text)

    def _validate(self, audio_file: str, binary: list):
        executable = binary[0] if binary else ""
        if not executable:
            raise ValueError("WHISPERX_BINARY cannot be empty.")
        if not Path(executable).exists() and shutil.which(executable) is None:
            raise ValueError(
                "WhisperX executable was not found. Set WHISPERX_BINARY to "
                "the whisperx command or install it with `pip install whisperx`."
            )
        if not os.path.exists(audio_file):
            raise ValueError(f"File {audio_file} does not exist.")

    @staticmethod
    def _read_txt_output(output_dir: str, audio_file: str) -> str:
        expected = Path(output_dir) / (Path(audio_file).stem + ".txt")
        if expected.exists():
            return expected.read_text(encoding="utf-8").strip()

        txt_outputs = sorted(Path(output_dir).glob("*.txt"))
        if txt_outputs:
            return txt_outputs[0].read_text(encoding="utf-8").strip()

        raise RuntimeError("WhisperX did not produce a .txt transcript.")

    @staticmethod
    def _is_truthy(value) -> bool:
        return str(value).lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _split_command(value: str) -> list:
        return shlex.split(value, posix=os.name != "nt")
