import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from sapat.script import main


class ScriptTests(unittest.TestCase):
    @patch("sapat.script.ReplicateTranscription")
    def test_replicate_api_choice_wires_replicate_transcriber(self, transcriber_cls):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "sample.mp4"
            video_path.write_bytes(b"fake video")
            transcriber = transcriber_cls.return_value

            result = CliRunner().invoke(main, [str(video_path), "--api", "replicate"])

        self.assertEqual(result.exit_code, 0)
        transcriber_cls.assert_called_once_with(temperature=0.3)
        transcriber.process_file.assert_called_once_with(
            video_path, "en", None, 0.3, "M", False
        )

    def test_replicate_api_rejects_correct_flag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "sample.mp4"
            video_path.write_bytes(b"fake video")

            result = CliRunner().invoke(
                main, [str(video_path), "--api", "replicate", "--correct"]
            )

        self.assertEqual(result.exit_code, 1)
        self.assertIn("Transcript correction is not supported", result.output)


if __name__ == "__main__":
    unittest.main()
