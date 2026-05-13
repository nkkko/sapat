import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from sapat.transcription.google import GoogleSpeechTranscription


class GoogleSpeechTranscriptionTest(unittest.TestCase):
    @patch.dict(os.environ, {"GOOGLE_SPEECH_API_KEY": "test-key"}, clear=False)
    @patch("sapat.transcription.google.requests.post")
    def test_transcribe_audio_posts_google_request_and_extracts_text(self, post_mock):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "results": [
                {"alternatives": [{"transcript": "first segment"}]},
                {"alternatives": [{"transcript": "second segment"}]},
            ]
        }
        post_mock.return_value = response

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as audio_file:
            audio_file.write(b"fake audio")
            audio_path = audio_file.name

        try:
            transcriber = GoogleSpeechTranscription(temperature=0.3)
            result = transcriber.transcribe_audio(
                audio_path,
                language="en",
                response_format="text",
            )
        finally:
            Path(audio_path).unlink(missing_ok=True)

        self.assertEqual(result, "first segment\nsecond segment")

        _, kwargs = post_mock.call_args
        self.assertEqual(kwargs["params"], {"key": "test-key"})
        self.assertEqual(kwargs["json"]["config"]["encoding"], "MP3")
        self.assertEqual(kwargs["json"]["config"]["languageCode"], "en-US")
        self.assertEqual(kwargs["json"]["config"]["model"], "latest_long")
        self.assertIn("content", kwargs["json"]["audio"])


if __name__ == "__main__":
    unittest.main()
