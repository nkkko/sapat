import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sapat.script import main


class ScriptLeopardRoutingTest(unittest.TestCase):
    def test_leopard_api_routes_to_picovoice_transcriber(self):
        calls = []

        class FakeTranscriber:
            def __init__(self, temperature):
                self.temperature = temperature

            def process_file(self, input_file, language, prompt, temperature, quality, correct):
                calls.append(
                    {
                        "input_file": str(input_file),
                        "language": language,
                        "prompt": prompt,
                        "temperature": temperature,
                        "quality": quality,
                        "correct": correct,
                    }
                )

        with tempfile.NamedTemporaryFile(suffix=".mp4") as video:
            with patch("sapat.script.PicovoiceLeopardTranscription", FakeTranscriber):
                result = CliRunner().invoke(
                    main,
                    [
                        video.name,
                        "--api",
                        "leopard",
                        "--language",
                        "en",
                        "--prompt",
                        "product names",
                        "--temperature",
                        "0.2",
                        "--quality",
                        "H",
                    ],
                )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["language"], "en")
        self.assertEqual(calls[0]["prompt"], "product names")
        self.assertEqual(calls[0]["temperature"], 0.2)
        self.assertEqual(calls[0]["quality"], "H")
        self.assertFalse(calls[0]["correct"])


if __name__ == "__main__":
    unittest.main()
