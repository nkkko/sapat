import unittest
from unittest.mock import Mock, patch

from click.testing import CliRunner

from sapat.script import main


class ScriptTests(unittest.TestCase):
    def test_cli_routes_lemonfox_provider(self):
        runner = CliRunner()
        transcriber = Mock()

        with runner.isolated_filesystem():
            with open("clip.mp4", "wb") as f:
                f.write(b"video")

            with patch("sapat.script.LemonfoxTranscription", return_value=transcriber) as constructor:
                result = runner.invoke(main, ["clip.mp4", "--api", "lemonfox"])

        self.assertEqual(result.exit_code, 0)
        constructor.assert_called_once_with(temperature=0.3)
        transcriber.process_file.assert_called_once()


if __name__ == "__main__":
    unittest.main()
