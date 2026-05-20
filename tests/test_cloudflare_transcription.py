import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from sapat.transcription.cloudflare import CloudflareTranscription


class CloudflareTranscriptionTest(unittest.TestCase):
    def setUp(self):
        self.previous_env = {
            "CLOUDFLARE_ACCOUNT_ID": os.environ.get("CLOUDFLARE_ACCOUNT_ID"),
            "CLOUDFLARE_API_TOKEN": os.environ.get("CLOUDFLARE_API_TOKEN"),
            "CLOUDFLARE_API_ENDPOINT": os.environ.get("CLOUDFLARE_API_ENDPOINT"),
            "CLOUDFLARE_WHISPER_MODEL": os.environ.get("CLOUDFLARE_WHISPER_MODEL"),
        }
        os.environ["CLOUDFLARE_ACCOUNT_ID"] = "account-123"
        os.environ["CLOUDFLARE_API_TOKEN"] = "token-abc"
        os.environ.pop("CLOUDFLARE_API_ENDPOINT", None)
        os.environ.pop("CLOUDFLARE_WHISPER_MODEL", None)

    def tearDown(self):
        for key, value in self.previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    @patch("sapat.transcription.cloudflare.requests.post")
    def test_transcribe_audio_posts_binary_audio_to_workers_ai(self, post):
        post.return_value = Mock(
            status_code=200,
            text="ok",
            json=Mock(return_value={"success": True, "result": {"text": "hello"}}),
        )

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as audio:
            audio.write(b"audio-bytes")
            path = audio.name

        try:
            result = CloudflareTranscription(temperature=0.3).transcribe_audio(path)
        finally:
            os.unlink(path)

        self.assertEqual({"text": "hello"}, result)
        post.assert_called_once_with(
            "https://api.cloudflare.com/client/v4/accounts/account-123/ai/run/@cf/openai/whisper",
            headers={
                "Authorization": "Bearer token-abc",
                "Content-Type": "audio/mpeg",
            },
            data=b"audio-bytes",
        )

    def test_missing_token_raises_actionable_error(self):
        os.environ.pop("CLOUDFLARE_API_TOKEN", None)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as audio:
            audio.write(b"audio-bytes")
            path = audio.name

        try:
            with self.assertRaisesRegex(ValueError, "CLOUDFLARE_API_TOKEN"):
                CloudflareTranscription(temperature=0.3).transcribe_audio(path)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
