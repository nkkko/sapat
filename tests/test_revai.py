import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


class RevAITranscriptionTests(unittest.TestCase):
    def test_transcribes_audio_by_polling_revai_job(self):
        from sapat.transcription.revai import RevAITranscription

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "meeting.mp3"
            audio_path.write_bytes(b"fake mp3 data")

            submit_response = Mock(status_code=200)
            submit_response.json.return_value = {"id": "job-123"}
            polling_response = Mock(status_code=200)
            polling_response.json.return_value = {"id": "job-123", "status": "transcribed"}
            transcript_response = Mock(status_code=200, text="Hello from Rev AI.")

            with patch.dict(
                "os.environ",
                {
                    "REVAI_ACCESS_TOKEN": "rev-token",
                    "REVAI_API_ENDPOINT": "https://api.rev.ai/speechtotext/v1",
                    "REVAI_TIMEOUT_SECONDS": "1",
                    "REVAI_POLL_INTERVAL_SECONDS": "0",
                },
                clear=False,
            ), patch("requests.post", return_value=submit_response) as post, patch(
                "requests.get", side_effect=[polling_response, transcript_response]
            ) as get:
                transcriber = RevAITranscription(temperature=0.3)

                result = transcriber.transcribe_audio(
                    str(audio_path),
                    language="en",
                    prompt="Daytona, Sapat",
                    temperature=0.1,
                )

        self.assertEqual(result, "Hello from Rev AI.")
        post.assert_called_once()
        post_args, post_kwargs = post.call_args
        self.assertEqual(post_args[0], "https://api.rev.ai/speechtotext/v1/jobs")
        self.assertEqual(
            post_kwargs["headers"]["Authorization"],
            "Bearer rev-token",
        )
        self.assertIn("media", post_kwargs["files"])
        self.assertIn("options", post_kwargs["data"])
        get.assert_any_call(
            "https://api.rev.ai/speechtotext/v1/jobs/job-123",
            headers={"Authorization": "Bearer rev-token"},
        )
        get.assert_any_call(
            "https://api.rev.ai/speechtotext/v1/jobs/job-123/transcript",
            headers={
                "Authorization": "Bearer rev-token",
                "Accept": "text/plain",
            },
        )

    def test_requires_access_token(self):
        from sapat.transcription.revai import RevAITranscription

        with patch.dict("os.environ", {"REVAI_ACCESS_TOKEN": ""}, clear=False):
            with self.assertRaisesRegex(ValueError, "REVAI_ACCESS_TOKEN"):
                RevAITranscription(temperature=0.3)

    def test_raises_when_job_fails(self):
        from sapat.transcription.revai import RevAITranscription

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "meeting.mp3"
            audio_path.write_bytes(b"fake mp3 data")

            submit_response = Mock(status_code=200)
            submit_response.json.return_value = {"id": "job-123"}
            polling_response = Mock(status_code=200)
            polling_response.json.return_value = {
                "id": "job-123",
                "status": "failed",
                "failure": "bad audio",
            }

            with patch.dict(
                "os.environ",
                {
                    "REVAI_ACCESS_TOKEN": "rev-token",
                    "REVAI_POLL_INTERVAL_SECONDS": "0",
                    "REVAI_TIMEOUT_SECONDS": "1",
                },
                clear=False,
            ), patch("requests.post", return_value=submit_response), patch(
                "requests.get", return_value=polling_response
            ):
                transcriber = RevAITranscription(temperature=0.3)

                with self.assertRaisesRegex(Exception, "bad audio"):
                    transcriber.transcribe_audio(str(audio_path))


if __name__ == "__main__":
    unittest.main()
