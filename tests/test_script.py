import tempfile
import unittest
from unittest.mock import patch

from click.testing import CliRunner

from sapat.script import main


class ScriptTest(unittest.TestCase):
    @patch("sapat.script.AssemblyAITranscription")
    def test_cli_routes_assemblyai_api_to_provider(self, provider):
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".mp4") as video:
            result = runner.invoke(main, [video.name, "--api", "assemblyai"])

        self.assertEqual(result.exit_code, 0)
        provider.assert_called_once_with(temperature=0.3)
        provider.return_value.process_file.assert_called_once()


if __name__ == "__main__":
    unittest.main()
