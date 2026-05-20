import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from click.testing import CliRunner

from sapat.script import main


class ScriptTest(unittest.TestCase):
    def test_help_lists_ibm_api_choice(self):
        result = CliRunner().invoke(main, ["--help"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("openai", result.output)
        self.assertIn("groq", result.output)
        self.assertIn("azure", result.output)
        self.assertIn("ibm", result.output)

    def test_ibm_api_choice_uses_ibm_watson_transcriber(self):
        transcriber = Mock()
        with tempfile.NamedTemporaryFile(suffix=".mp4") as input_file:
            with patch("sapat.script.IBMWatsonTranscription", return_value=transcriber) as transcriber_class:
                result = CliRunner().invoke(
                    main,
                    [
                        input_file.name,
                        "--api",
                        "ibm",
                        "--language",
                        "en",
                        "--quality",
                        "H",
                    ],
                )

        self.assertEqual(result.exit_code, 0)
        transcriber_class.assert_called_once_with(temperature=0.3)
        transcriber.process_file.assert_called_once_with(
            Path(input_file.name),
            "en",
            None,
            0.3,
            "H",
            False,
        )


if __name__ == "__main__":
    unittest.main()
