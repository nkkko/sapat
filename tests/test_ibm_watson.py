import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from sapat.transcription.ibm_watson import IBMWatsonTranscription


class IBMWatsonTranscriptionTest(unittest.TestCase):
    def test_transcribe_audio_posts_audio_and_extracts_transcript(self):
        response = Mock(ok=True)
        response.json.return_value = {
            "results": [
                {
                    "alternatives": [
                        {"transcript": "first segment "}
                    ]
                },
                {
                    "alternatives": [
                        {"transcript": "second segment "}
                    ]
                },
            ]
        }

        env = {
            "IBM_WATSON_STT_API_KEY": "test-key",
            "IBM_WATSON_STT_URL": "https://example.watson.cloud.ibm.com/instances/test",
            "IBM_WATSON_STT_MODEL": "en-US_BroadbandModel",
            "IBM_WATSON_STT_SMART_FORMATTING": "true",
        }

        with tempfile.NamedTemporaryFile(suffix=".mp3") as audio_file:
            with patch.dict(os.environ, env):
                with patch("sapat.transcription.ibm_watson.requests.post", return_value=response) as post:
                    transcriber = IBMWatsonTranscription(temperature=0.3)
                    transcript = transcriber.transcribe_audio(audio_file.name)

        self.assertEqual(transcript, "first segment second segment")
        post.assert_called_once()
        _, kwargs = post.call_args
        self.assertEqual(kwargs["auth"], ("apikey", "test-key"))
        self.assertEqual(kwargs["headers"], {"Content-Type": "audio/mpeg"})
        self.assertEqual(
            kwargs["params"],
            {"model": "en-US_BroadbandModel", "smart_formatting": "true"},
        )

    def test_requires_api_key_and_service_url(self):
        with patch.dict(os.environ, {}, clear=True):
            transcriber = IBMWatsonTranscription(temperature=0.3)

        with self.assertRaisesRegex(ValueError, "IBM_WATSON_STT_API_KEY"):
            transcriber._validate_config()


if __name__ == "__main__":
    unittest.main()
