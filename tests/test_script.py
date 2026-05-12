import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from click.testing import CliRunner

from sapat.script import main


class ScriptTests(unittest.TestCase):
    def test_routes_revai_api_choice_to_revai_transcriber(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "clip.mp4"
            video_path.write_bytes(b"fake video")
            transcriber = Mock()

            with patch("sapat.script.RevAITranscription", return_value=transcriber):
                result = CliRunner().invoke(
                    main,
                    [str(video_path), "--api", "revai"],
                )

        self.assertEqual(result.exit_code, 0)
        transcriber.process_file.assert_called_once_with(
            video_path,
            "en",
            None,
            0.3,
            "M",
            False,
        )


if __name__ == "__main__":
    unittest.main()
