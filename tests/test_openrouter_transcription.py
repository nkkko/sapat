import base64
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from sapat.transcription.openrouter import OpenRouterTranscription


class OpenRouterTranscriptionTest(unittest.TestCase):
    def setUp(self):
        self.original_env = os.environ.copy()
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        os.environ["OPENROUTER_MODEL"] = "openai/whisper-large-v3"

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.original_env)

    @patch("sapat.transcription.openrouter.requests.post")
    def test_transcribe_audio_posts_base64_payload(self, post):
        response = Mock(status_code=200)
        response.json.return_value = {"text": "hello from openrouter"}
        post.return_value = response

        with tempfile.NamedTemporaryFile(suffix=".mp3") as audio:
            audio.write(b"audio bytes")
            audio.flush()

            result = OpenRouterTranscription(temperature=0.2).transcribe_audio(
                audio.name,
                language="en",
                temperature=0.4,
            )

        self.assertEqual(result, {"text": "hello from openrouter"})
        post.assert_called_once()
        _, kwargs = post.call_args
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(kwargs["headers"]["Content-Type"], "application/json")
        self.assertEqual(kwargs["json"]["model"], "openai/whisper-large-v3")
        self.assertEqual(kwargs["json"]["language"], "en")
        self.assertEqual(kwargs["json"]["temperature"], 0.4)
        self.assertEqual(kwargs["json"]["input_audio"]["format"], "mp3")
        self.assertEqual(
            kwargs["json"]["input_audio"]["data"],
            base64.b64encode(b"audio bytes").decode("ascii"),
        )

    def test_missing_api_key_raises_clear_error(self):
        os.environ.pop("OPENROUTER_API_KEY")

        with tempfile.NamedTemporaryFile(suffix=".mp3") as audio:
            with self.assertRaisesRegex(ValueError, "OPENROUTER_API_KEY"):
                OpenRouterTranscription(temperature=0.2).transcribe_audio(audio.name)

    def test_unsupported_extension_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".txt") as audio:
            with self.assertRaisesRegex(ValueError, "Unsupported audio file format"):
                OpenRouterTranscription(temperature=0.2).transcribe_audio(audio.name)


if __name__ == "__main__":
    unittest.main()
