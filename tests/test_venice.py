import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from sapat.transcription.venice import VeniceTranscription


class VeniceTranscriptionTests(unittest.TestCase):
    def test_transcribe_audio_posts_expected_multipart_payload(self):
        with tempfile.NamedTemporaryFile(suffix=".mp3") as audio:
            audio.write(b"audio")
            audio.flush()

            fake_response = Mock()
            fake_response.status_code = 200
            fake_response.json.return_value = {"text": "hello"}

            env = {
                "VENICE_API_KEY": "test-key",
                "VENICE_MODEL": "openai/whisper-large-v3",
                "VENICE_API_ENDPOINT": "https://api.venice.ai/api/v1/audio/transcriptions",
                "VENICE_TIMESTAMPS": "true",
            }
            with patch.dict(os.environ, env), patch(
                "sapat.transcription.venice.requests.post", return_value=fake_response
            ) as post:
                transcriber = VeniceTranscription(temperature=0.3)
                result = transcriber.transcribe_audio(audio.name, language="en")

            self.assertEqual(result, {"text": "hello"})
            args, kwargs = post.call_args
            self.assertEqual(args[0], "https://api.venice.ai/api/v1/audio/transcriptions")
            self.assertEqual(kwargs["headers"], {"Authorization": "Bearer test-key"})
            self.assertEqual(
                kwargs["data"],
                {
                    "model": "openai/whisper-large-v3",
                    "response_format": "json",
                    "timestamps": "true",
                    "language": "en",
                },
            )
            self.assertIn("file", kwargs["files"])

    def test_transcribe_audio_requires_api_key(self):
        with tempfile.NamedTemporaryFile(suffix=".mp3") as audio, patch.dict(
            os.environ, {"VENICE_API_KEY": ""}, clear=True
        ):
            transcriber = VeniceTranscription(temperature=0.3)
            with self.assertRaises(ValueError):
                transcriber.transcribe_audio(audio.name)


if __name__ == "__main__":
    unittest.main()
