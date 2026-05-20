import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from sapat.transcription.together import TogetherAITranscription


class TogetherAITranscriptionTest(unittest.TestCase):
    def setUp(self):
        self.original_env = os.environ.copy()
        os.environ["TOGETHER_API_KEY"] = "test-key"
        os.environ["TOGETHER_MODEL"] = "openai/whisper-large-v3"
        os.environ["TOGETHER_API_ENDPOINT"] = "https://api.together.xyz/v1/audio/transcriptions"
        self.audio_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        self.audio_file.write(b"audio")
        self.audio_file.close()

    def tearDown(self):
        os.unlink(self.audio_file.name)
        os.environ.clear()
        os.environ.update(self.original_env)

    @patch("sapat.transcription.together.requests.post")
    def test_transcribe_audio_posts_multipart_request(self, post):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"text": "hello from together"}
        post.return_value = response

        result = TogetherAITranscription(temperature=0.2).transcribe_audio(
            self.audio_file.name,
            language="en",
            prompt="product names",
        )

        self.assertEqual(result, "hello from together")
        post.assert_called_once()
        _, kwargs = post.call_args
        self.assertEqual(post.call_args[0][0], "https://api.together.xyz/v1/audio/transcriptions")
        self.assertEqual(kwargs["headers"], {"Authorization": "Bearer test-key"})
        self.assertEqual(kwargs["data"]["model"], "openai/whisper-large-v3")
        self.assertEqual(kwargs["data"]["response_format"], "json")
        self.assertEqual(kwargs["data"]["temperature"], 0.2)
        self.assertEqual(kwargs["data"]["language"], "en")
        self.assertEqual(kwargs["data"]["prompt"], "product names")
        self.assertEqual(kwargs["files"]["file"][0], os.path.basename(self.audio_file.name))
        self.assertEqual(kwargs["files"]["file"][2], "audio/mpeg")

    def test_missing_api_key_raises_clear_error(self):
        del os.environ["TOGETHER_API_KEY"]
        transcriber = TogetherAITranscription(temperature=0.3)

        with self.assertRaisesRegex(ValueError, "TOGETHER_API_KEY"):
            transcriber.transcribe_audio(self.audio_file.name)


if __name__ == "__main__":
    unittest.main()
