import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from sapat.transcription.gladia import GladiaTranscription


class GladiaTranscriptionTest(unittest.TestCase):
    def test_transcribe_audio_uploads_creates_job_and_polls_result(self):
        upload_response = Mock(ok=True)
        upload_response.json.return_value = {"audio_url": "https://api.gladia.io/file/audio-id"}

        create_response = Mock(ok=True)
        create_response.json.return_value = {
            "id": "job-id",
            "result_url": "https://api.gladia.io/v2/pre-recorded/job-id",
        }

        queued_response = Mock(ok=True)
        queued_response.json.return_value = {"status": "queued"}

        done_response = Mock(ok=True)
        done_response.json.return_value = {
            "status": "done",
            "result": {
                "transcription": {
                    "full_transcript": "Gladia transcript",
                }
            },
        }

        with tempfile.NamedTemporaryFile(suffix=".mp3") as audio_file:
            with patch.dict(os.environ, {"GLADIA_API_KEY": "test-key"}):
                with patch("sapat.transcription.gladia.requests.post", side_effect=[upload_response, create_response]) as post:
                    with patch("sapat.transcription.gladia.requests.get", side_effect=[queued_response, done_response]) as get:
                        with patch("sapat.transcription.gladia.time.sleep"):
                            transcriber = GladiaTranscription(temperature=0.3, poll_interval=0.01, timeout=1)
                            transcript = transcriber.transcribe_audio(audio_file.name, language="en")

        self.assertEqual(transcript, "Gladia transcript")
        self.assertEqual(set(post.call_args_list[0][1]["files"].keys()), {"audio"})
        self.assertEqual(
            post.call_args_list[1][1]["json"]["language_config"],
            {"languages": ["en"], "code_switching": False},
        )
        get.assert_called_with("https://api.gladia.io/v2/pre-recorded/job-id", headers={"x-gladia-key": "test-key"})

    def test_requires_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            transcriber = GladiaTranscription(temperature=0.3)

        with self.assertRaisesRegex(ValueError, "GLADIA_API_KEY"):
            transcriber._headers()


if __name__ == "__main__":
    unittest.main()
