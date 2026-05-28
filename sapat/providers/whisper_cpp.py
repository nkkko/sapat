# ABOUTME: whisper.cpp local transcription provider
# ABOUTME: Uses whisper-cli or main binary for offline speech recognition

import os
import re
import shlex
import shutil
import subprocess
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
class WhisperCppProvider(TranscriptionProvider):
    name = "whisper_cpp"
    config = ProviderConfig(
        required_env_vars=[],
        required_packages=[],
        max_file_size_mb=float("inf"),
        preferred_format=AudioFormat.WAV,
        supports_correction=False,
        default_model="base",
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
        binary = os.getenv("WHISPER_CPP_BINARY", "whisper-cli")
        model_path = os.getenv("WHISPER_CPP_MODEL_PATH")
        threads = os.getenv("WHISPER_CPP_THREADS")
        extra_args = shlex.split(os.getenv("WHISPER_CPP_EXTRA_ARGS", ""))

        self._validate(audio_file, binary, model_path)

        command = [
            binary,
            "-m", model_path,
            "-f", audio_file,
            "-nt",
        ]

        if language:
            command.extend(["-l", language])
        if prompt:
            command.extend(["--prompt", prompt])
        if threads:
            command.extend(["-t", threads])

        command.extend(extra_args)

        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )

        text = self._clean_output(result.stdout)
        return TranscriptionResult(text=text)

    def _validate(self, audio_file: str, binary: str, model_path: Optional[str]):
        if not model_path:
            raise ValueError(
                "WHISPER_CPP_MODEL_PATH must point to a local ggml model file."
            )
        if not Path(model_path).is_file():
            raise ValueError(f"whisper.cpp model not found: {model_path}")
        if not Path(audio_file).is_file():
            raise ValueError(f"Audio file not found: {audio_file}")

        binary_path = Path(binary)
        if not binary_path.is_file() and shutil.which(binary) is None:
            raise ValueError(
                "whisper.cpp binary not found. Set WHISPER_CPP_BINARY to "
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
