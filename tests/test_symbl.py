import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from sapat.transcription.symbl import SymblTranscription


class SymblTranscriptionTests(unittest.TestCase):
    def setUp(self):
        self._env = patch.dict(
            os.environ,
            {
                "SYMBL_ACCESS_TOKEN": "test-token",
                "SYMBL_API_BASE_URL": "https://symbl.test/v1",
            },
            clear=False,
        )
        self._env.start()

    def tearDown(self):
        self._env.stop()

    def _audio_file(self):
        temp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        temp.write(b"audio")
        temp.close()
        self.addCleanup(lambda: Path(temp.name).exists() and Path(temp.name).unlink())
        return temp.name

    @staticmethod
    def _response(status_code, payload=None, text=""):
        response = Mock()
        response.status_code = status_code
        response.text = text
        response.json.return_value = payload or {}
        return response

    @patch("sapat.transcription.symbl.requests.get")
    @patch("sapat.transcription.symbl.requests.post")
    def test_transcribe_audio_uses_async_job_and_messages(self, mock_post, mock_get):
        audio_file = self._audio_file()
        mock_post.return_value = self._response(
            201,
            {
                "jobId": "job-123",
                "conversationId": "conversation-456",
            },
        )
        mock_get.side_effect = [
            self._response(200, {"status": "in_progress"}),
            self._response(200, {"status": "completed"}),
            self._response(
                200,
                {
                    "messages": [
                        {"text": "First sentence."},
                        {"text": "Second sentence."},
                    ],
                },
            ),
        ]

        transcriber = SymblTranscription(temperature=0.3, poll_interval_seconds=0)
        result = transcriber.transcribe_audio(audio_file, language="en")

        self.assertEqual(result, {"text": "First sentence.\nSecond sentence."})
        mock_post.assert_called_once()
        post_kwargs = mock_post.call_args.kwargs
        self.assertEqual(post_kwargs["headers"]["Authorization"], "Bearer test-token")
        self.assertEqual(post_kwargs["headers"]["Content-Type"], "audio/mpeg")
        self.assertEqual(post_kwargs["params"], {"languageCode": "en-US"})
        self.assertEqual(mock_get.call_args_list[-1].args[0], "https://symbl.test/v1/conversations/conversation-456/messages")

    @patch("sapat.transcription.symbl.requests.post")
    def test_generates_access_token_when_missing(self, mock_post):
        with patch.dict(
            os.environ,
            {
                "SYMBL_ACCESS_TOKEN": "",
                "SYMBL_APP_ID": "app-id",
                "SYMBL_APP_SECRET": "app-secret",
            },
            clear=False,
        ):
            mock_post.return_value = self._response(200, {"accessToken": "generated-token"})
            transcriber = SymblTranscription(temperature=0.3)

            self.assertEqual(transcriber._get_access_token(), "generated-token")
            mock_post.assert_called_once_with(
                "https://api.symbl.ai/oauth2/token:generate",
                json={
                    "type": "application",
                    "appId": "app-id",
                    "appSecret": "app-secret",
                },
                timeout=30,
            )

    @patch("sapat.transcription.symbl.requests.get")
    @patch("sapat.transcription.symbl.requests.post")
    def test_failed_job_raises_error(self, mock_post, mock_get):
        audio_file = self._audio_file()
        mock_post.return_value = self._response(201, {"jobId": "job-123", "conversationId": "conversation-456"})
        mock_get.return_value = self._response(200, {"status": "failed", "message": "bad audio"})

        transcriber = SymblTranscription(temperature=0.3, poll_interval_seconds=0)

        with self.assertRaisesRegex(Exception, "Symbl job failed"):
            transcriber.transcribe_audio(audio_file, language="en")


if __name__ == "__main__":
    unittest.main()
