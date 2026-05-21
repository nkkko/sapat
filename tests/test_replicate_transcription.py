import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sapat.transcription.replicate import ReplicateTranscription


class ReplicateTranscriptionTests(unittest.TestCase):
    def setUp(self):
        self.env_patch = patch.dict(
            os.environ,
            {
                "REPLICATE_API_TOKEN": "test-token",
                "REPLICATE_MODEL": "openai/whisper",
            },
            clear=False,
        )
        self.env_patch.start()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.audio_path = Path(self.tmpdir.name) / "sample.mp3"
        self.audio_path.write_bytes(b"fake audio")

    def tearDown(self):
        self.tmpdir.cleanup()
        self.env_patch.stop()

    @patch("sapat.transcription.replicate.replicate.Client")
    def test_transcribe_audio_returns_text_from_replicate_transcription(self, client_cls):
        run = client_cls.return_value.run
        run.return_value = {"transcription": "hello from replicate"}

        result = ReplicateTranscription(temperature=0.3).transcribe_audio(str(self.audio_path))

        self.assertEqual(result, {"text": "hello from replicate"})
        client_cls.assert_called_once_with(api_token="test-token")
        self.assertEqual(run.call_args.args[0], "openai/whisper")
        self.assertEqual(run.call_args.kwargs["input"]["audio"].name, str(self.audio_path))

    @patch("sapat.transcription.replicate.replicate.Client")
    def test_transcribe_audio_includes_translate_when_enabled(self, client_cls):
        run = client_cls.return_value.run
        run.return_value = {"text": "translated text"}

        ReplicateTranscription(temperature=0.3).transcribe_audio(
            str(self.audio_path), translate=True
        )

        self.assertTrue(run.call_args.kwargs["input"]["translate"])

    @patch("sapat.transcription.replicate.replicate.Client")
    def test_transcribe_audio_includes_optional_whisper_model(self, client_cls):
        run = client_cls.return_value.run
        run.return_value = {"text": "text"}

        with patch.dict(os.environ, {"REPLICATE_WHISPER_MODEL": "large-v3"}, clear=False):
            ReplicateTranscription(temperature=0.3).transcribe_audio(str(self.audio_path))

        self.assertEqual(run.call_args.kwargs["input"]["model"], "large-v3")

    def test_extracts_segment_text_when_output_contains_segments(self):
        transcriber = ReplicateTranscription(temperature=0.3)

        text = transcriber._extract_transcription_text(
            {"segments": [{"text": "hello"}, {"text": " world"}]}
        )

        self.assertEqual(text, "hello world")

    def test_missing_token_raises_clear_error(self):
        with patch.dict(os.environ, {"REPLICATE_API_TOKEN": ""}, clear=False):
            transcriber = ReplicateTranscription(temperature=0.3)

        with self.assertRaisesRegex(ValueError, "REPLICATE_API_TOKEN"):
            transcriber.transcribe_audio(str(self.audio_path))

    def test_rejects_files_over_configured_limit(self):
        with patch.dict(os.environ, {"REPLICATE_MAX_FILE_SIZE_MB": "0"}, clear=False):
            transcriber = ReplicateTranscription(temperature=0.3)

        with self.assertRaisesRegex(ValueError, "upload limit"):
            transcriber.transcribe_audio(str(self.audio_path))

    def test_rejects_unsupported_audio_extension(self):
        bad_path = Path(self.tmpdir.name) / "sample.txt"
        bad_path.write_text("not audio")
        transcriber = ReplicateTranscription(temperature=0.3)

        with self.assertRaisesRegex(ValueError, "Unsupported audio file format"):
            transcriber.transcribe_audio(str(bad_path))

    def test_unsupported_output_shape_raises_clear_error(self):
        transcriber = ReplicateTranscription(temperature=0.3)

        with self.assertRaisesRegex(ValueError, "Unsupported Replicate transcription output"):
            transcriber._extract_transcription_text({"segments": [{"timestamp": [0, 1]}]})


if __name__ == "__main__":
    unittest.main()
