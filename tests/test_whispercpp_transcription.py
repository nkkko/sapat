import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from sapat.script import main
from sapat.transcription.whispercpp import WhisperCppTranscription


class WhisperCppTranscriptionTest(unittest.TestCase):
    def setUp(self):
        self.audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        self.audio.write(b"audio")
        self.audio.close()
        self.model = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
        self.model.write(b"model")
        self.model.close()

    def tearDown(self):
        Path(self.audio.name).unlink(missing_ok=True)
        Path(self.model.name).unlink(missing_ok=True)

    def _env(self, **extra):
        env = {
            "WHISPER_CPP_BINARY": "custom-whisper-cli",
            "WHISPER_CPP_MODEL_PATH": self.model.name,
        }
        env.update(extra)
        return patch.dict(os.environ, env, clear=False)

    def test_runs_whispercpp_and_reads_generated_text_output(self):
        commands = []

        def fake_run(command, **kwargs):
            commands.append(command)
            if command[0] == "custom-whisper-cli":
                output_base = Path(command[command.index("-of") + 1])
                output_base.with_suffix(".txt").write_text(
                    "hello from local whisper",
                    encoding="utf-8",
                )
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        with self._env(WHISPER_CPP_THREADS="6"):
            with patch(
                "sapat.transcription.whispercpp.subprocess.run",
                side_effect=fake_run,
            ):
                result = WhisperCppTranscription(temperature=0.2).transcribe_audio(
                    self.audio.name,
                    language="en",
                    prompt="Product names: Sapat, Daytona",
                )

        self.assertEqual(result, "hello from local whisper")
        self.assertEqual(commands[0][0], "ffmpeg")
        whisper_command = commands[1]
        self.assertEqual(whisper_command[0], "custom-whisper-cli")
        self.assertIn("-m", whisper_command)
        self.assertIn(self.model.name, whisper_command)
        self.assertIn("-f", whisper_command)
        self.assertIn("-otxt", whisper_command)
        self.assertIn("-nt", whisper_command)
        self.assertIn("-l", whisper_command)
        self.assertIn("en", whisper_command)
        self.assertIn("--prompt", whisper_command)
        self.assertIn("Product names: Sapat, Daytona", whisper_command)
        self.assertIn("-t", whisper_command)
        self.assertIn("6", whisper_command)

    def test_uses_stdout_when_cli_does_not_write_output_file(self):
        def fake_run(command, **kwargs):
            if command[0] == "custom-whisper-cli":
                return subprocess.CompletedProcess(command, 0, stdout="stdout transcript", stderr="")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        with self._env():
            with patch(
                "sapat.transcription.whispercpp.subprocess.run",
                side_effect=fake_run,
            ):
                result = WhisperCppTranscription(temperature=0.2).transcribe_audio(
                    self.audio.name
                )

        self.assertEqual(result, "stdout transcript")

    def test_requires_model_path(self):
        with patch.dict(os.environ, {"WHISPER_CPP_MODEL_PATH": ""}, clear=False):
            with self.assertRaisesRegex(ValueError, "WHISPER_CPP_MODEL_PATH"):
                WhisperCppTranscription(temperature=0.2).transcribe_audio(self.audio.name)

    @patch("sapat.script.WhisperCppTranscription")
    def test_cli_wires_whispercpp_provider(self, mock_transcriber_cls):
        runner = CliRunner()
        mock_transcriber = MagicMock()
        mock_transcriber_cls.return_value = mock_transcriber

        with tempfile.NamedTemporaryFile(suffix=".mp4") as video_file:
            result = runner.invoke(main, [video_file.name, "--api", "whispercpp"])

        self.assertEqual(result.exit_code, 0)
        mock_transcriber_cls.assert_called_once_with(temperature=0.3)
        mock_transcriber.process_file.assert_called_once()


if __name__ == "__main__":
    unittest.main()
