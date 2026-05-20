import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from sapat.transcription.lemonfox import LemonfoxTranscription


class LemonfoxTranscriptionTests(unittest.TestCase):
    def test_transcribe_audio_posts_openai_compatible_form(self):
        with tempfile.NamedTemporaryFile(suffix=".mp3") as audio_file:
            audio_file.write(b"audio")
            audio_file.flush()

            response = Mock()
            response.status_code = 200
            response.json.return_value = {"text": "hello world"}

            with patch.dict(os.environ, {"LEMONFOX_API_KEY": "test-key"}):
                with patch("sapat.transcription.lemonfox.requests.post", return_value=response) as post:
                    transcriber = LemonfoxTranscription(temperature=0.3)
                    result = transcriber.transcribe_audio(
                        audio_file.name,
                        language="english",
                        prompt="Daytona, Sapat",
                    )

            self.assertEqual(result, {"text": "hello world"})
            post.assert_called_once()
            _, kwargs = post.call_args
            self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-key")
            self.assertEqual(
                kwargs["data"],
                {
                    "response_format": "json",
                    "language": "english",
                    "prompt": "Daytona, Sapat",
                },
            )
            self.assertIn("file", kwargs["files"])

    def test_transcribe_audio_requires_api_key(self):
        with tempfile.NamedTemporaryFile(suffix=".mp3") as audio_file:
            audio_file.write(b"audio")
            audio_file.flush()

            with patch.dict(os.environ, {"LEMONFOX_API_KEY": ""}):
                transcriber = LemonfoxTranscription(temperature=0.3)
                with self.assertRaisesRegex(ValueError, "LEMONFOX_API_KEY"):
                    transcriber.transcribe_audio(audio_file.name)


if __name__ == "__main__":
    unittest.main()
