import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from sapat.script import main
from sapat.transcription.deepinfra import DeepInfraTranscription


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self.payload = payload
        self.text = text

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class DeepInfraTranscriptionTest(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def test_requires_token(self):
        audio_file = self._audio_file(".mp3")
        transcriber = DeepInfraTranscription(temperature=0.3)

        with self.assertRaisesRegex(ValueError, "DEEPINFRA_TOKEN"):
            transcriber.transcribe_audio(audio_file)

    @patch("sapat.transcription.deepinfra.requests.post")
    def test_posts_audio_to_default_model_endpoint(self, mock_post):
        os.environ["DEEPINFRA_TOKEN"] = "test-token"
        os.environ["DEEPINFRA_MODEL"] = "openai/whisper-small"
        mock_post.return_value = FakeResponse(payload={"text": "hello world"})
        audio_file = self._audio_file(".mp3")

        result = DeepInfraTranscription(temperature=0.2).transcribe_audio(
            audio_file,
            language="en",
            prompt="Product names: Sapat",
        )

        self.assertEqual(result["text"], "hello world")
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        self.assertEqual(
            kwargs["headers"]["Authorization"],
            "Bearer test-token",
        )
        self.assertEqual(
            mock_post.call_args.args[0],
            "https://api.deepinfra.com/v1/inference/openai/whisper-small",
        )
        self.assertEqual(kwargs["data"]["language"], "en")
        self.assertEqual(kwargs["data"]["prompt"], "Product names: Sapat")
        self.assertEqual(kwargs["data"]["temperature"], 0.2)
        self.assertIn("audio", kwargs["files"])

    @patch("sapat.transcription.deepinfra.requests.post")
    def test_allows_endpoint_override(self, mock_post):
        os.environ["DEEPINFRA_TOKEN"] = "test-token"
        os.environ["DEEPINFRA_API_ENDPOINT"] = (
            "https://proxy.example/v1/inference/openai/whisper-large"
        )
        mock_post.return_value = FakeResponse(payload={"text": "via proxy"})
        audio_file = self._audio_file(".wav")

        result = DeepInfraTranscription(temperature=0.1).transcribe_audio(audio_file)

        self.assertEqual(result["text"], "via proxy")
        self.assertEqual(
            mock_post.call_args.args[0],
            "https://proxy.example/v1/inference/openai/whisper-large",
        )

    @patch("sapat.transcription.deepinfra.requests.post")
    def test_raises_api_errors(self, mock_post):
        os.environ["DEEPINFRA_TOKEN"] = "test-token"
        mock_post.return_value = FakeResponse(status_code=401, text="unauthorized")
        audio_file = self._audio_file(".mp3")

        with self.assertRaisesRegex(Exception, "DeepInfra transcription failed"):
            DeepInfraTranscription(temperature=0.3).transcribe_audio(audio_file)

    def test_rejects_unsupported_audio_extension(self):
        os.environ["DEEPINFRA_TOKEN"] = "test-token"
        audio_file = self._audio_file(".flac")

        with self.assertRaisesRegex(ValueError, "Unsupported audio file format"):
            DeepInfraTranscription(temperature=0.3).transcribe_audio(audio_file)

    @patch("sapat.script.DeepInfraTranscription")
    def test_cli_routes_deepinfra_choice(self, mock_transcription):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_file = Path(tmpdir) / "demo.mp4"
            video_file.write_bytes(b"fake video")
            transcriber = MagicMock()
            mock_transcription.return_value = transcriber

            result = CliRunner().invoke(
                main,
                [str(video_file), "--api", "deepinfra", "--temperature", "0.4"],
            )

        self.assertEqual(result.exit_code, 0)
        mock_transcription.assert_called_once_with(temperature=0.4)
        transcriber.process_file.assert_called_once()

    def _audio_file(self, suffix):
        handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        handle.write(b"audio")
        handle.close()
        self.addCleanup(self._remove_file, handle.name)
        return handle.name

    @staticmethod
    def _remove_file(file_name):
        path = Path(file_name)
        if path.exists():
            path.unlink()


if __name__ == "__main__":
    unittest.main()
