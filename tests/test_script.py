import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from sapat.script import main


class ScriptTests(unittest.TestCase):
    def test_oracle_api_routes_to_oracle_transcriber(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("input.mp4").write_bytes(b"fake video")
            with patch("sapat.script.OracleSpeechTranscription") as transcriber_cls:
                result = runner.invoke(main, ["input.mp4", "--api", "oracle"])

        self.assertEqual(result.exit_code, 0)
        transcriber_cls.assert_called_once()
        transcriber_cls.return_value.process_file.assert_called_once()


if __name__ == "__main__":
    unittest.main()
