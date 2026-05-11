import os
import tempfile
import unittest
from unittest.mock import ANY, Mock, patch

from sapat.transcription.assemblyai import AssemblyAITranscription


class AssemblyAITranscriptionTest(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(
            os.environ,
            {
                "ASSEMBLYAI_API_KEY": "test-key",
                "ASSEMBLYAI_POLL_INTERVAL_SECONDS": "0",
                "ASSEMBLYAI_TIMEOUT_SECONDS": "5",
            },
            clear=False,
        )
        self.env.start()
        self.audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        self.audio.write(b"fake audio")
        self.audio.close()

    def tearDown(self):
        self.env.stop()
        os.unlink(self.audio.name)

    @patch("sapat.transcription.assemblyai.requests.get")
    @patch("sapat.transcription.assemblyai.requests.post")
    def test_transcribes_uploaded_audio(self, post, get):
        upload_response = Mock(status_code=200)
        upload_response.json.return_value = {
            "upload_url": "https://cdn.example/audio.mp3"
        }
        submit_response = Mock(status_code=200)
        submit_response.json.return_value = {"id": "transcript-123"}
        post.side_effect = [upload_response, submit_response]

        poll_response = Mock(status_code=200)
        poll_response.json.return_value = {
            "status": "completed",
            "text": "Hello from Daytona",
        }
        get.return_value = poll_response

        transcriber = AssemblyAITranscription(temperature=0.3)
        result = transcriber.transcribe_audio(self.audio.name, language="en")

        self.assertEqual(result["text"], "Hello from Daytona")
        self.assertEqual(result["id"], "transcript-123")
        post.assert_any_call(
            "https://api.assemblyai.com/v2/transcript",
            headers={"Authorization": "test-key", "Content-Type": "application/json"},
            json={"audio_url": "https://cdn.example/audio.mp3", "language_code": "en"},
        )

    @patch("sapat.transcription.assemblyai.requests.get")
    @patch("sapat.transcription.assemblyai.requests.post")
    def test_returns_plain_text_when_requested(self, post, get):
        upload_response = Mock(status_code=200)
        upload_response.json.return_value = {
            "upload_url": "https://cdn.example/audio.mp3"
        }
        submit_response = Mock(status_code=200)
        submit_response.json.return_value = {"id": "transcript-123"}
        post.side_effect = [upload_response, submit_response]

        poll_response = Mock(status_code=200)
        poll_response.json.return_value = {
            "status": "completed",
            "text": "plain transcript",
        }
        get.return_value = poll_response

        transcriber = AssemblyAITranscription(temperature=0.3, response_format="text")

        self.assertEqual(
            transcriber.transcribe_audio(self.audio.name),
            "plain transcript",
        )

    @patch("sapat.transcription.assemblyai.requests.get")
    @patch("sapat.transcription.assemblyai.requests.post")
    def test_raises_when_transcript_job_errors(self, post, get):
        upload_response = Mock(status_code=200)
        upload_response.json.return_value = {
            "upload_url": "https://cdn.example/audio.mp3"
        }
        submit_response = Mock(status_code=200)
        submit_response.json.return_value = {"id": "transcript-123"}
        post.side_effect = [upload_response, submit_response]

        poll_response = Mock(status_code=200)
        poll_response.json.return_value = {"status": "error", "error": "bad audio"}
        get.return_value = poll_response

        transcriber = AssemblyAITranscription(temperature=0.3)
        with self.assertRaisesRegex(Exception, "bad audio"):
            transcriber.transcribe_audio(self.audio.name)

    @patch("sapat.transcription.assemblyai.requests.post")
    def test_raises_when_api_key_is_missing(self, post):
        with patch.dict(os.environ, {"ASSEMBLYAI_API_KEY": ""}, clear=False):
            transcriber = AssemblyAITranscription(temperature=0.3)

        with self.assertRaisesRegex(ValueError, "ASSEMBLYAI_API_KEY"):
            transcriber.transcribe_audio(self.audio.name)
        post.assert_not_called()

    @patch("sapat.transcription.assemblyai.requests.get")
    @patch("sapat.transcription.assemblyai.requests.post")
    def test_times_out_when_transcript_remains_queued(self, post, get):
        upload_response = Mock(status_code=200)
        upload_response.json.return_value = {
            "upload_url": "https://cdn.example/audio.mp3"
        }
        submit_response = Mock(status_code=200)
        submit_response.json.return_value = {"id": "transcript-123"}
        post.side_effect = [upload_response, submit_response]

        poll_response = Mock(status_code=200)
        poll_response.json.return_value = {"status": "queued"}
        get.return_value = poll_response

        transcriber = AssemblyAITranscription(
            temperature=0.3,
            poll_interval_seconds=0,
            timeout_seconds=0,
        )

        with self.assertRaisesRegex(TimeoutError, "timed out"):
            transcriber.transcribe_audio(self.audio.name)

    @patch("sapat.transcription.assemblyai.requests.post")
    def test_upload_http_errors_include_status_code(self, post):
        upload_response = Mock(status_code=401, text="unauthorized")
        post.return_value = upload_response

        transcriber = AssemblyAITranscription(temperature=0.3)

        with self.assertRaisesRegex(RuntimeError, "401"):
            transcriber.transcribe_audio(self.audio.name)

    def test_accepts_uppercase_supported_extensions(self):
        audio = tempfile.NamedTemporaryFile(suffix=".WAV", delete=False)
        audio.write(b"fake audio")
        audio.close()
        self.addCleanup(os.unlink, audio.name)

        transcriber = AssemblyAITranscription(temperature=0.3)

        transcriber._validate_audio_file(audio.name)

    @patch("sapat.transcription.assemblyai.requests.post")
    def test_strips_trailing_slash_from_custom_endpoint(self, post):
        upload_response = Mock(status_code=200)
        upload_response.json.return_value = {
            "upload_url": "https://cdn.example/audio.mp3"
        }
        post.return_value = upload_response

        with patch.dict(
            os.environ,
            {"ASSEMBLYAI_API_ENDPOINT": "https://example.test/v2/"},
            clear=False,
        ):
            transcriber = AssemblyAITranscription(temperature=0.3)

        self.assertEqual(
            transcriber._upload_audio(self.audio.name),
            "https://cdn.example/audio.mp3",
        )
        post.assert_called_once_with(
            "https://example.test/v2/upload",
            headers={"Authorization": "test-key"},
            data=ANY,
        )

    def test_requires_openai_settings_for_correction(self):
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "", "OPENAI_MODEL_NAME_CHAT": ""},
            clear=False,
        ):
            transcriber = AssemblyAITranscription(temperature=0.3)

        with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
            transcriber.generate_corrected_transcript(
                self.audio.name,
                temperature=0.7,
                system_prompt="Fix punctuation.",
            )


if __name__ == "__main__":
    unittest.main()
