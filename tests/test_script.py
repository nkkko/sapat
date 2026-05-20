import tempfile
import unittest
from unittest.mock import patch

from click.testing import CliRunner

from sapat.script import main


class ScriptTest(unittest.TestCase):
    @patch("sapat.script.TogetherAITranscription")
    def test_together_api_routes_to_together_transcriber(self, together):
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".mp4") as video:
            result = runner.invoke(main, [video.name, "--api", "together"])

        self.assertEqual(result.exit_code, 0)
        together.assert_called_once_with(temperature=0.3)
        together.return_value.process_file.assert_called_once()


if __name__ == "__main__":
    unittest.main()
