import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from click.testing import CliRunner

from sapat.script import main
from sapat.transcription.fireworks import FireworksTranscription


class FireworksTranscriptionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        self.tmp.write(b"audio")
        self.tmp.close()

    def tearDown(self):
        os.unlink(self.tmp.name)

    @patch.dict(os.environ, {"FIREWORKS_API_KEY": "fw-key"}, clear=True)
    @patch("sapat.transcription.fireworks.requests.post")
    def test_posts_to_default_audio_prod_endpoint(self, post):
        post.return_value = Mock(status_code=200, json=lambda: {"text": "hello"})

        result = FireworksTranscription(temperature=0.2).transcribe_audio(
            self.tmp.name,
            language="en",
            prompt="Use product names exactly.",
        )

        self.assertEqual(result, {"text": "hello"})
        args, kwargs = post.call_args
        self.assertEqual(args[0], "https://audio-prod.api.fireworks.ai/v1/audio/transcriptions")
        self.assertEqual(kwargs["headers"], {"Authorization": "fw-key"})
        self.assertEqual(kwargs["data"]["model"], "whisper-v3")
        self.assertEqual(kwargs["data"]["language"], "en")
        self.assertEqual(kwargs["data"]["prompt"], "Use product names exactly.")
        self.assertEqual(kwargs["data"]["temperature"], 0.2)
        self.assertIn("file", kwargs["files"])

    @patch.dict(os.environ, {"FIREWORKS_API_KEY": "fw-key", "FIREWORKS_MODEL": "whisper-v3-turbo"}, clear=True)
    @patch("sapat.transcription.fireworks.requests.post")
    def test_turbo_model_uses_turbo_endpoint(self, post):
        post.return_value = Mock(status_code=200, json=lambda: {"text": "fast"})

        FireworksTranscription(temperature=0.3).transcribe_audio(self.tmp.name)

        args, _kwargs = post.call_args
        self.assertEqual(args[0], "https://audio-turbo.api.fireworks.ai/v1/audio/transcriptions")

    @patch.dict(
        os.environ,
        {
            "FIREWORKS_API_KEY": "fw-key",
            "FIREWORKS_API_ENDPOINT": "https://example.test/v1/audio/transcriptions",
            "FIREWORKS_PREPROCESSING": "soft_dynamic",
        },
        clear=True,
    )
    @patch("sapat.transcription.fireworks.requests.post")
    def test_custom_endpoint_and_optional_settings(self, post):
        post.return_value = Mock(status_code=200, json=lambda: {"text": "custom"})

        FireworksTranscription(temperature=0.1).transcribe_audio(self.tmp.name)

        args, kwargs = post.call_args
        self.assertEqual(args[0], "https://example.test/v1/audio/transcriptions")
        self.assertEqual(kwargs["data"]["preprocessing"], "soft_dynamic")

    @patch.dict(os.environ, {}, clear=True)
    def test_requires_api_key(self):
        with self.assertRaisesRegex(ValueError, "FIREWORKS_API_KEY"):
            FireworksTranscription(temperature=0.2).transcribe_audio(self.tmp.name)

    def test_correction_is_explicitly_unsupported(self):
        with self.assertRaisesRegex(NotImplementedError, "Correction is not implemented"):
            FireworksTranscription(temperature=0.2).generate_corrected_transcript(self.tmp.name, 0.7, "Fix it")

    @patch("sapat.script.FireworksTranscription")
    def test_cli_routes_fireworks_provider(self, fireworks_cls):
        transcriber = fireworks_cls.return_value
        runner = CliRunner()
        with runner.isolated_filesystem():
            with open("demo.mp4", "wb") as f:
                f.write(b"video")

            result = runner.invoke(main, ["demo.mp4", "--api", "fireworks"])

        self.assertEqual(result.exit_code, 0)
        fireworks_cls.assert_called_once_with(temperature=0.3)
        transcriber.process_file.assert_called_once()


if __name__ == "__main__":
    unittest.main()
