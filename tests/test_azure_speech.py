import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class AzureSpeechTranscriptionTests(unittest.TestCase):
    def _write_audio_file(self):
        handle = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        handle.write(b"fake audio")
        handle.close()
        self.addCleanup(lambda: Path(handle.name).exists() and Path(handle.name).unlink())
        return handle.name

    @patch.dict(os.environ, {
        "AZURE_SPEECH_API_KEY": "test-key",
        "AZURE_SPEECH_REGION": "eastus",
    }, clear=True)
    @patch("sapat.transcription.azure_speech.requests.post")
    def test_transcribe_audio_posts_fast_transcription_request(self, post):
        from sapat.transcription.azure_speech import AzureSpeechTranscription

        response = Mock(status_code=200)
        response.json.return_value = {
            "combinedPhrases": [
                {"text": "Hola."},
                {"text": "Mundo."},
            ]
        }
        post.return_value = response

        result = AzureSpeechTranscription(temperature=0.1).transcribe_audio(
            self._write_audio_file(),
            language="es",
        )

        self.assertEqual(result, "Hola.\nMundo.")
        post.assert_called_once()
        url = post.call_args.args[0]
        kwargs = post.call_args.kwargs
        self.assertEqual(
            url,
            "https://eastus.api.cognitive.microsoft.com/speechtotext/transcriptions:transcribe?api-version=2025-10-15",
        )
        self.assertEqual(kwargs["headers"]["Ocp-Apim-Subscription-Key"], "test-key")
        self.assertEqual(json.loads(kwargs["data"]["definition"]), {"locales": ["es-ES"]})
        self.assertIn("audio", kwargs["files"])

    @patch.dict(os.environ, {
        "AZURE_SPEECH_KEY": "fallback-key",
        "AZURE_SPEECH_ENDPOINT": "https://speech.example.cognitiveservices.azure.com/",
        "AZURE_SPEECH_API_VERSION": "2025-10-15",
    }, clear=True)
    @patch("sapat.transcription.azure_speech.requests.post")
    def test_transcribe_audio_uses_endpoint_override_and_combined_phrase_fallback(self, post):
        from sapat.transcription.azure_speech import AzureSpeechTranscription

        response = Mock(status_code=200)
        response.json.return_value = {"combinedPhrases": [{"text": "Custom endpoint."}]}
        post.return_value = response

        result = AzureSpeechTranscription(temperature=0.1).transcribe_audio(
            self._write_audio_file(),
            language="en-US",
        )

        self.assertEqual(result, "Custom endpoint.")
        self.assertEqual(
            post.call_args.args[0],
            "https://speech.example.cognitiveservices.azure.com/speechtotext/transcriptions:transcribe?api-version=2025-10-15",
        )
        self.assertEqual(post.call_args.kwargs["headers"]["Ocp-Apim-Subscription-Key"], "fallback-key")
        self.assertEqual(
            json.loads(post.call_args.kwargs["data"]["definition"]),
            {"locales": ["en-US"]},
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_transcribe_audio_requires_key_and_endpoint_or_region(self):
        from sapat.transcription.azure_speech import AzureSpeechTranscription

        with self.assertRaises(ValueError) as context:
            AzureSpeechTranscription(temperature=0.1).transcribe_audio(self._write_audio_file())

        self.assertIn("AZURE_SPEECH", str(context.exception))


class AzureSpeechCliTests(unittest.TestCase):
    def test_cli_help_lists_azure_speech_provider(self):
        from click.testing import CliRunner
        from sapat.script import main

        result = CliRunner().invoke(main, ["--help"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("azurespeech", result.output)


if __name__ == "__main__":
    unittest.main()
