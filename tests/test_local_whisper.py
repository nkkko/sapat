import os
import tempfile
import unittest
from types import SimpleNamespace

from sapat.transcription.local_whisper import LocalWhisperTranscription


class FakeModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio_file, **kwargs):
        self.calls.append((audio_file, kwargs))
        segments = [
            SimpleNamespace(text=" Hello Daytona "),
            SimpleNamespace(text=""),
            SimpleNamespace(text=" local transcription."),
        ]
        return segments, SimpleNamespace(language="en", duration=4.2)


class LocalWhisperTranscriptionTest(unittest.TestCase):
    def test_transcribe_audio_returns_json_response(self):
        model = FakeModel()
        transcriber = LocalWhisperTranscription(
            temperature=0.2,
            model_factory=lambda: model,
        )

        with tempfile.NamedTemporaryFile(suffix=".mp3") as audio:
            result = transcriber.transcribe_audio(
                audio.name,
                language="en",
                prompt="Product terms: Daytona, Sapat",
                temperature=0.1,
            )

        self.assertEqual(result["text"], "Hello Daytona local transcription.")
        self.assertEqual(result["language"], "en")
        self.assertEqual(result["duration"], 4.2)
        self.assertEqual(model.calls[0][1]["language"], "en")
        self.assertEqual(model.calls[0][1]["initial_prompt"], "Product terms: Daytona, Sapat")
        self.assertEqual(model.calls[0][1]["temperature"], 0.1)

    def test_transcribe_audio_can_return_plain_text(self):
        model = FakeModel()
        transcriber = LocalWhisperTranscription(
            temperature=0.2,
            response_format="text",
            model_factory=lambda: model,
        )

        with tempfile.NamedTemporaryFile(suffix=".mp3") as audio:
            result = transcriber.transcribe_audio(audio.name)

        self.assertEqual(result, "Hello Daytona local transcription.")

    def test_invalid_extension_is_rejected(self):
        transcriber = LocalWhisperTranscription(
            temperature=0.2,
            model_factory=lambda: FakeModel(),
        )

        with tempfile.NamedTemporaryFile(suffix=".txt") as audio:
            with self.assertRaises(ValueError):
                transcriber.transcribe_audio(audio.name)

    def test_environment_config_is_used_for_model_loading(self):
        old_env = dict(os.environ)
        os.environ["SAPAT_LOCAL_WHISPER_MODEL"] = "small"
        os.environ["SAPAT_LOCAL_WHISPER_DEVICE"] = "cuda"
        os.environ["SAPAT_LOCAL_WHISPER_COMPUTE_TYPE"] = "float16"
        os.environ["SAPAT_LOCAL_WHISPER_BEAM_SIZE"] = "3"
        try:
            transcriber = LocalWhisperTranscription(
                temperature=0.2,
                model_factory=lambda: FakeModel(),
            )
            self.assertEqual(transcriber.model_name, "small")
            self.assertEqual(transcriber.device, "cuda")
            self.assertEqual(transcriber.compute_type, "float16")
            self.assertEqual(transcriber.beam_size, 3)
        finally:
            os.environ.clear()
            os.environ.update(old_env)


if __name__ == "__main__":
    unittest.main()
