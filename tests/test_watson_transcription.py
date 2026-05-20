import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from sapat.transcription.watson import IBMWatsonTranscription


class IBMWatsonTranscriptionTest(unittest.TestCase):
    def setUp(self):
        self.original_env = os.environ.copy()
        os.environ["IBM_WATSON_STT_API_KEY"] = "test-key"
        os.environ["IBM_WATSON_STT_URL"] = "https://api.us-south.speech-to-text.watson.cloud.ibm.com/instances/test"
        os.environ["IBM_WATSON_STT_MODEL"] = "en-US_BroadbandModel"
        self.audio_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        self.audio_file.write(b"audio")
        self.audio_file.close()

    def tearDown(self):
        os.unlink(self.audio_file.name)
        os.environ.clear()
        os.environ.update(self.original_env)

    @patch("sapat.transcription.watson.requests.post")
    def test_transcribe_audio_posts_to_recognize_endpoint(self, post):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "results": [
                {"alternatives": [{"transcript": "hello world "}]},
                {"alternatives": [{"transcript": "from watson"}]},
            ]
        }
        post.return_value = response

        result = IBMWatsonTranscription(temperature=0.3).transcribe_audio(self.audio_file.name)

        self.assertEqual(result, "hello world from watson")
        post.assert_called_once()
        _, kwargs = post.call_args
        self.assertEqual(
            post.call_args[0][0],
            "https://api.us-south.speech-to-text.watson.cloud.ibm.com/instances/test/v1/recognize",
        )
        self.assertEqual(kwargs["auth"], ("apikey", "test-key"))
        self.assertEqual(kwargs["headers"], {"Content-Type": "audio/mpeg"})
        self.assertEqual(kwargs["params"], {"model": "en-US_BroadbandModel"})

    def test_missing_api_key_raises_clear_error(self):
        del os.environ["IBM_WATSON_STT_API_KEY"]
        transcriber = IBMWatsonTranscription(temperature=0.3)

        with self.assertRaisesRegex(ValueError, "IBM_WATSON_STT_API_KEY"):
            transcriber.transcribe_audio(self.audio_file.name)

    def test_recognize_url_accepts_full_endpoint(self):
        os.environ["IBM_WATSON_STT_URL"] = (
            "https://api.us-south.speech-to-text.watson.cloud.ibm.com/instances/test/v1/recognize"
        )

        self.assertEqual(
            IBMWatsonTranscription(temperature=0.3)._recognize_url(),
            "https://api.us-south.speech-to-text.watson.cloud.ibm.com/instances/test/v1/recognize",
        )


if __name__ == "__main__":
    unittest.main()
