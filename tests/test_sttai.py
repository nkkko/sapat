import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from sapat.transcription.sttai import STTAITranscription


class STTAITranscriptionTests(unittest.TestCase):
    def setUp(self):
        self.env = {
            "STTAI_API_KEY": "test-key",
            "STTAI_API_ENDPOINT": "https://api.stt.ai/v1/transcribe",
            "STTAI_MODEL": "small",
            "STTAI_DIARIZE": "false",
            "STTAI_SPEAKERS": "1",
        }

    def test_transcribe_audio_posts_file_and_options(self):
        response = Mock(status_code=200)
        response.json.return_value = {"text": "hello world"}

        with tempfile.NamedTemporaryFile(suffix=".mp3") as audio_file, patch.dict(os.environ, self.env, clear=False), patch(
            "sapat.transcription.sttai.requests.post", return_value=response
        ) as post:
            audio_file.write(b"fake audio")
            audio_file.flush()

            result = STTAITranscription(temperature=0.3).transcribe_audio(audio_file.name, language="es")

        self.assertEqual(result, {"text": "hello world"})
        args, kwargs = post.call_args
        self.assertEqual(args[0], "https://api.stt.ai/v1/transcribe")
        self.assertEqual(kwargs["headers"], {"Authorization": "Bearer test-key"})
        self.assertEqual(kwargs["data"]["model"], "small")
        self.assertEqual(kwargs["data"]["language"], "es")
        self.assertEqual(kwargs["data"]["diarize"], "false")
        self.assertEqual(kwargs["data"]["speakers"], "1")
        self.assertEqual(kwargs["data"]["response_format"], "json")
        self.assertEqual(kwargs["files"]["file"][0], os.path.basename(audio_file.name))

    def test_transcribe_audio_allows_anonymous_requests(self):
        response = Mock(status_code=200, text="plain transcript")

        with tempfile.NamedTemporaryFile(suffix=".mp3") as audio_file, patch.dict(os.environ, {"STTAI_API_KEY": ""}, clear=False), patch(
            "sapat.transcription.sttai.requests.post", return_value=response
        ) as post:
            audio_file.write(b"fake audio")
            audio_file.flush()

            result = STTAITranscription(temperature=0.3, response_format="txt").transcribe_audio(audio_file.name)

        self.assertEqual(result, "plain transcript")
        self.assertEqual(post.call_args.kwargs["headers"], {})

    def test_transcribe_audio_rejects_missing_file(self):
        with self.assertRaises(ValueError):
            STTAITranscription(temperature=0.3).transcribe_audio("/tmp/not-here.mp3")


if __name__ == "__main__":
    unittest.main()
