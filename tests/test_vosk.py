import os
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from sapat.transcription.vosk import VoskTranscription


class FakeRecognizer:
    def __init__(self, model, sample_rate):
        self.model = model
        self.sample_rate = sample_rate

    def AcceptWaveform(self, data):
        return True

    def Result(self):
        return '{"text": "hello"}'

    def FinalResult(self):
        return '{"text": "world"}'


class VoskTranscriptionTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.model_dir = Path(self.tmpdir.name) / "model"
        self.model_dir.mkdir()
        self.audio_file = Path(self.tmpdir.name) / "input.mp3"
        self.audio_file.write_bytes(b"fake audio")
        self.wav_file = Path(self.tmpdir.name) / "input.wav"
        with wave.open(str(self.wav_file), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(b"\x00\x00" * 16)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_transcribes_with_mocked_vosk(self):
        fake_vosk = types.SimpleNamespace(
            Model=lambda model_path: {"model_path": model_path},
            KaldiRecognizer=FakeRecognizer,
        )
        original_vosk = sys.modules.get("vosk")
        sys.modules["vosk"] = fake_vosk
        self.addCleanup(self._restore_vosk_module, original_vosk)

        with patch.dict(os.environ, {"VOSK_MODEL_PATH": str(self.model_dir)}):
            transcriber = VoskTranscription()
            with patch.object(transcriber, "_convert_to_wav", return_value=str(self.wav_file)):
                result = transcriber.transcribe_audio(str(self.audio_file))

        self.assertEqual(result, {"text": "hello world"})
        self.assertFalse(self.wav_file.exists())

    def test_requires_model_path(self):
        with patch.dict(os.environ, {}, clear=True):
            transcriber = VoskTranscription()
            with self.assertRaisesRegex(ValueError, "VOSK_MODEL_PATH"):
                transcriber.transcribe_audio(str(self.audio_file))

    @staticmethod
    def _restore_vosk_module(original_vosk):
        if original_vosk is None:
            sys.modules.pop("vosk", None)
        else:
            sys.modules["vosk"] = original_vosk
