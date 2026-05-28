import tempfile
import unittest
from unittest.mock import patch

from click.testing import CliRunner

from sapat.script import main


class ScriptTest(unittest.TestCase):
    @patch("sapat.script.OpenRouterTranscription")
    def test_openrouter_cli_route(self, transcriber_class):
        transcriber = transcriber_class.return_value

        with tempfile.NamedTemporaryFile(suffix=".mp4") as video:
            result = CliRunner().invoke(main, [video.name, "--api", "openrouter"])

        self.assertEqual(result.exit_code, 0)
        transcriber_class.assert_called_once_with(temperature=0.3)
        transcriber.process_file.assert_called_once()


if __name__ == "__main__":
    unittest.main()
