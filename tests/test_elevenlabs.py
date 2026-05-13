import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from sapat.transcription.elevenlabs import ElevenLabsTranscription


class ElevenLabsTranscriptionTests(unittest.TestCase):
    def test_transcribe_audio_sends_expected_multipart_request(self):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"text": "Hello from Sapat"}

        with tempfile.NamedTemporaryFile(suffix=".mp3") as audio:
            audio.write(b"fake audio")
            audio.flush()

            with patch.dict(
                os.environ,
                {
                    "ELEVENLABS_API_KEY": "test-key",
                    "ELEVENLABS_MODEL": "scribe_v2",
                    "ELEVENLABS_API_ENDPOINT": "https://example.test/speech-to-text",
                },
            ):
                transcriber = ElevenLabsTranscription(temperature=0.2)

            with patch(
                "sapat.transcription.elevenlabs.requests.post",
                return_value=response,
            ) as post:
                result = transcriber.transcribe_audio(audio.name, language="en")

        self.assertEqual(result, {"text": "Hello from Sapat"})
        post.assert_called_once()

        _, kwargs = post.call_args
        self.assertEqual(kwargs["headers"], {"xi-api-key": "test-key"})
        self.assertEqual(
            kwargs["data"],
            {
                "model_id": "scribe_v2",
                "temperature": 0.2,
                "language_code": "en",
            },
        )
        self.assertEqual(kwargs["timeout"], 120)
        self.assertIn("file", kwargs["files"])

    def test_validate_audio_file_requires_api_key(self):
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": ""}):
            transcriber = ElevenLabsTranscription(temperature=0.2)

        with tempfile.NamedTemporaryFile(suffix=".mp3") as audio:
            with self.assertRaisesRegex(ValueError, "ELEVENLABS_API_KEY"):
                transcriber.transcribe_audio(audio.name)

    def test_validate_audio_file_rejects_unsupported_extension(self):
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test-key"}):
            transcriber = ElevenLabsTranscription(temperature=0.2)

        with tempfile.NamedTemporaryFile(suffix=".txt") as audio:
            with self.assertRaisesRegex(ValueError, "Unsupported audio file"):
                transcriber.transcribe_audio(audio.name)


if __name__ == "__main__":
    unittest.main()
