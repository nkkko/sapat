import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from sapat.script import main
from sapat.transcription.whispercpp import WhisperCppTranscription


class WhisperCppTranscriptionTests(unittest.TestCase):
    def test_clean_output_removes_timestamps_and_runtime_lines(self):
        output = """
whisper_init_from_file: loading model
[00:00:00.000 --> 00:00:01.000] Hello Daytona.
main: processing done
Plain line
"""

        self.assertEqual(
            WhisperCppTranscription._clean_output(output),
            "Hello Daytona.\nPlain line",
        )

    def test_missing_model_path_is_reported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio = Path(tmpdir) / "sample.wav"
            audio.write_bytes(b"audio")

            with patch.dict(os.environ, {}, clear=True):
                transcriber = WhisperCppTranscription()

            with self.assertRaisesRegex(ValueError, "WHISPERCPP_MODEL_PATH"):
                transcriber.transcribe_audio(str(audio))

    def test_transcribe_audio_builds_whispercpp_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = Path(tmpdir) / "ggml-base.en.bin"
            audio = Path(tmpdir) / "sample.wav"
            binary = Path(tmpdir) / "whisper-cli"
            model.write_bytes(b"model")
            audio.write_bytes(b"audio")
            binary.write_text("#!/bin/sh\n", encoding="utf-8")

            completed = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="[00:00:00.000 --> 00:00:01.000] Local transcript",
                stderr="",
            )

            with patch.dict(
                os.environ,
                {
                    "WHISPERCPP_MODEL_PATH": str(model),
                    "WHISPERCPP_BINARY": str(binary),
                    "WHISPERCPP_THREADS": "4",
                    "WHISPERCPP_EXTRA_ARGS": "--no-prints",
                },
                clear=True,
            ):
                transcriber = WhisperCppTranscription()

            with patch(
                "sapat.transcription.whispercpp.subprocess.run",
                return_value=completed,
            ) as run:
                transcript = transcriber.transcribe_audio(
                    str(audio),
                    language="en",
                    prompt="Daytona workspace terms",
                )

            self.assertEqual(transcript, "Local transcript")
            self.assertEqual(
                run.call_args.args[0],
                [
                    str(binary),
                    "-m",
                    str(model),
                    "-f",
                    str(audio),
                    "-nt",
                    "-l",
                    "en",
                    "--prompt",
                    "Daytona workspace terms",
                    "-t",
                    "4",
                    "--no-prints",
                ],
            )

    def test_process_file_converts_to_wav_and_writes_transcript(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = Path(tmpdir) / "ggml-base.en.bin"
            binary = Path(tmpdir) / "whisper-cli"
            video = Path(tmpdir) / "demo.mp4"
            output = Path(tmpdir) / "demo.txt"
            model.write_bytes(b"model")
            binary.write_text("#!/bin/sh\n", encoding="utf-8")
            video.write_bytes(b"video")

            def fake_run(command, **kwargs):
                if command[0] == "ffmpeg":
                    Path(command[-1]).write_bytes(b"wav")
                    return subprocess.CompletedProcess(command, 0)
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout="Offline transcript",
                    stderr="",
                )

            with patch.dict(
                os.environ,
                {
                    "WHISPERCPP_MODEL_PATH": str(model),
                    "WHISPERCPP_BINARY": str(binary),
                },
                clear=True,
            ):
                transcriber = WhisperCppTranscription()

            with patch(
                "sapat.transcription.whispercpp.subprocess.run",
                side_effect=fake_run,
            ):
                transcriber.process_file(video, "en", None, 0.3, "M", False)

            self.assertEqual(output.read_text(encoding="utf-8"), "Offline transcript")

    def test_cli_routes_whispercpp_api(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video = Path(tmpdir) / "demo.mp4"
            video.write_bytes(b"video")

            with patch.object(
                WhisperCppTranscription,
                "process_file",
                autospec=True,
            ) as process_file:
                result = CliRunner().invoke(
                    main,
                    [str(video), "--api", "whispercpp", "--language", "en"],
                )

            self.assertEqual(result.exit_code, 0)
            process_file.assert_called_once()


if __name__ == "__main__":
    unittest.main()
