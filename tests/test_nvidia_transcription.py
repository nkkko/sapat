import os
import tempfile
import unittest
from unittest import mock

from sapat.transcription.nvidia import NvidiaTranscription


class NvidiaTranscriptionTests(unittest.TestCase):
    def test_requires_api_key(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            transcriber = NvidiaTranscription(temperature=0.3)

        with tempfile.NamedTemporaryFile(suffix=".mp3") as audio:
            with self.assertRaisesRegex(ValueError, "NVIDIA_NIM_API_KEY"):
                transcriber.transcribe_audio(audio.name)

    def test_posts_audio_to_nim_endpoint(self):
        response = mock.Mock(status_code=200)
        response.json.return_value = {"text": "hello from parakeet"}

        with mock.patch.dict(
            os.environ,
            {
                "NVIDIA_NIM_API_KEY": "test-key",
                "NVIDIA_NIM_BASE_URL": "https://example.test/v1/",
                "NVIDIA_NIM_MODEL": "nvidia/parakeet-ctc-0.6b-asr",
            },
            clear=True,
        ):
            transcriber = NvidiaTranscription(temperature=0.3)

        with tempfile.NamedTemporaryFile(suffix=".mp3") as audio:
            audio.write(b"audio")
            audio.flush()
            with mock.patch("sapat.transcription.nvidia.requests.post", return_value=response) as post:
                result = transcriber.transcribe_audio(audio.name, language="en")

        self.assertEqual(result, {"text": "hello from parakeet"})
        post.assert_called_once()
        _, kwargs = post.call_args
        self.assertEqual(post.call_args.args[0], "https://example.test/v1/audio/transcriptions")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(kwargs["data"]["model"], "nvidia/parakeet-ctc-0.6b-asr")
        self.assertEqual(kwargs["data"]["language"], "en")
        self.assertIn("file", kwargs["files"])

    def test_rejects_unsupported_audio_extension(self):
        with mock.patch.dict(os.environ, {"NVIDIA_NIM_API_KEY": "test-key"}, clear=True):
            transcriber = NvidiaTranscription(temperature=0.3)

        with tempfile.NamedTemporaryFile(suffix=".txt") as audio:
            with self.assertRaisesRegex(ValueError, "Unsupported audio file format"):
                transcriber.transcribe_audio(audio.name)


if __name__ == "__main__":
    unittest.main()
