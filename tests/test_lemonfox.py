import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from sapat.transcription.lemonfox import LemonfoxTranscription


class LemonfoxTranscriptionTest(unittest.TestCase):
    def setUp(self):
        self.env = {
            "LEMONFOX_API_KEY": "test-key",
            "LEMONFOX_API_ENDPOINT": "https://api.lemonfox.ai/v1/audio/transcriptions",
        }

    def _audio_file(self, suffix=".mp3"):
        handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        handle.write(b"fake audio")
        handle.close()
        self.addCleanup(lambda: os.path.exists(handle.name) and os.unlink(handle.name))
        return handle.name

    @patch.dict(os.environ, {}, clear=True)
    def test_requires_api_key(self):
        transcriber = LemonfoxTranscription(temperature=0.3)
        with self.assertRaisesRegex(ValueError, "LEMONFOX_API_KEY"):
            transcriber.transcribe_audio(self._audio_file())

    @patch("sapat.transcription.lemonfox.requests.post")
    def test_posts_audio_to_lemonfox(self, post):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"text": "hello world"}
        post.return_value = response

        with patch.dict(os.environ, self.env, clear=True):
            transcriber = LemonfoxTranscription(temperature=0.3)
            result = transcriber.transcribe_audio(
                self._audio_file(),
                language="en",
                prompt="Daytona workspace",
            )

        self.assertEqual(result, {"text": "hello world"})
        _, kwargs = post.call_args
        self.assertEqual(
            kwargs["headers"]["Authorization"],
            "Bearer test-key",
        )
        self.assertEqual(kwargs["data"]["response_format"], "json")
        self.assertEqual(kwargs["data"]["language"], "english")
        self.assertEqual(kwargs["data"]["prompt"], "Daytona workspace")
        self.assertIn("file", kwargs["files"])
        self.assertEqual(kwargs["timeout"], 120)

    @patch("sapat.transcription.lemonfox.requests.post")
    def test_speaker_labels_switch_json_to_verbose_json(self, post):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"text": "hello", "segments": []}
        post.return_value = response

        env = {
            **self.env,
            "LEMONFOX_SPEAKER_LABELS": "true",
        }
        with patch.dict(os.environ, env, clear=True):
            transcriber = LemonfoxTranscription(temperature=0.3)
            transcriber.transcribe_audio(self._audio_file())

        _, kwargs = post.call_args
        self.assertEqual(kwargs["data"]["speaker_labels"], "true")
        self.assertEqual(kwargs["data"]["response_format"], "verbose_json")

    @patch("sapat.transcription.lemonfox.requests.post")
    def test_text_response_format_returns_body_text(self, post):
        response = MagicMock()
        response.status_code = 200
        response.text = "plain transcript"
        post.return_value = response

        env = {
            **self.env,
            "LEMONFOX_RESPONSE_FORMAT": "text",
        }
        with patch.dict(os.environ, env, clear=True):
            transcriber = LemonfoxTranscription(temperature=0.3)
            result = transcriber.transcribe_audio(self._audio_file())

        self.assertEqual(result, "plain transcript")

    def test_rejects_unsupported_audio_extension(self):
        with patch.dict(os.environ, self.env, clear=True):
            transcriber = LemonfoxTranscription(temperature=0.3)
            with self.assertRaisesRegex(ValueError, "Unsupported audio file format"):
                transcriber.transcribe_audio(self._audio_file(".txt"))


if __name__ == "__main__":
    unittest.main()
