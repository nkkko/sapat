import os
import tempfile
import unittest
from unittest import mock

from sapat.script import main
from sapat.transcription.sarvam import SarvamTranscription


class SarvamTranscriptionTests(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(
            os.environ,
            {
                "SARVAM_API_KEY": "test-key",
                "SARVAM_STT_ENDPOINT": "https://api.sarvam.ai/speech-to-text",
                "SARVAM_STT_MODEL": "saaras:v3",
                "SARVAM_STT_MODE": "transcribe",
            },
            clear=False,
        )
        self.env.start()

    def tearDown(self):
        self.env.stop()

    @mock.patch("sapat.transcription.sarvam.requests.post")
    def test_posts_sarvam_request_and_exposes_text_alias(self, post):
        post.return_value.status_code = 200
        post.return_value.json.return_value = {
            "request_id": "request-123",
            "transcript": "namaste",
            "language_code": "hi-IN",
        }

        with tempfile.NamedTemporaryFile(suffix=".mp3") as audio:
            result = SarvamTranscription(temperature=0.3).transcribe_audio(
                audio.name,
                language="hi",
            )

        self.assertEqual(result["text"], "namaste")
        _, kwargs = post.call_args
        self.assertEqual(kwargs["headers"], {"api-subscription-key": "test-key"})
        self.assertEqual(kwargs["data"]["model"], "saaras:v3")
        self.assertEqual(kwargs["data"]["mode"], "transcribe")
        self.assertEqual(kwargs["data"]["language_code"], "hi-IN")

    def test_missing_api_key_fails_before_request(self):
        with mock.patch.dict(os.environ, {"SARVAM_API_KEY": ""}, clear=False):
            with tempfile.NamedTemporaryFile(suffix=".mp3") as audio:
                with self.assertRaisesRegex(ValueError, "SARVAM_API_KEY"):
                    SarvamTranscription(temperature=0.3).transcribe_audio(audio.name)

    def test_cli_language_overrides_env_default(self):
        with mock.patch.dict(os.environ, {"SARVAM_LANGUAGE_CODE": "unknown"}, clear=False):
            transcriber = SarvamTranscription(temperature=0.3)
            self.assertEqual(transcriber._language_code("ta"), "ta-IN")

    def test_blank_optional_env_uses_endpoint_default(self):
        with mock.patch.dict(
            os.environ,
            {
                "SARVAM_STT_ENDPOINT": "",
                "SARVAM_STT_MODEL": "",
                "SARVAM_STT_MODE": "",
                "SARVAM_LANGUAGE_CODE": "",
            },
            clear=False,
        ):
            transcriber = SarvamTranscription(temperature=0.3)
            self.assertEqual(transcriber.endpoint, "https://api.sarvam.ai/speech-to-text")
            self.assertEqual(transcriber.model, "saaras:v3")
            self.assertEqual(transcriber.mode, "transcribe")
            self.assertIsNone(transcriber.default_language_code)

    def test_cli_accepts_sarvam_api_choice(self):
        runner = main.make_context(
            "sapat",
            ["sample.mp4", "--api", "sarvam"],
            resilient_parsing=True,
        )
        self.assertEqual(runner.params["api"], "sarvam")


if __name__ == "__main__":
    unittest.main()
