import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sapat.transcription.leopard import PicovoiceLeopardTranscription


class FakeLeopard:
    def __init__(self):
        self.deleted = False

    def process_file(self, audio_file):
        self.audio_file = audio_file
        return "hello from leopard", []

    def delete(self):
        self.deleted = True


class PicovoiceLeopardTranscriptionTest(unittest.TestCase):
    def test_transcribes_with_configured_engine_and_deletes_it(self):
        created = {}
        fake_engine = FakeLeopard()

        def factory(**kwargs):
            created.update(kwargs)
            return fake_engine

        env = {
            "PICOVOICE_ACCESS_KEY": "access-key",
            "PICOVOICE_LEOPARD_MODEL_PATH": "/models/leopard.pv",
            "PICOVOICE_LEOPARD_DEVICE": "best",
            "PICOVOICE_LEOPARD_ENABLE_PUNCTUATION": "true",
            "PICOVOICE_LEOPARD_ENABLE_DIARIZATION": "false",
        }

        with patch.dict(os.environ, env, clear=True):
            transcriber = PicovoiceLeopardTranscription(
                temperature=0.3, leopard_factory=factory
            )
            with tempfile.NamedTemporaryFile(suffix=".mp3") as audio:
                result = transcriber.transcribe_audio(audio.name)

        self.assertEqual(result, "hello from leopard")
        self.assertTrue(fake_engine.deleted)
        self.assertEqual(
            created,
            {
                "access_key": "access-key",
                "model_path": "/models/leopard.pv",
                "device": "best",
                "enable_automatic_punctuation": True,
                "enable_diarization": False,
            },
        )

    def test_missing_access_key_raises_clear_error(self):
        with patch.dict(os.environ, {}, clear=True):
            transcriber = PicovoiceLeopardTranscription(
                temperature=0.3, leopard_factory=lambda **kwargs: FakeLeopard()
            )
            with tempfile.NamedTemporaryFile(suffix=".mp3") as audio:
                with self.assertRaisesRegex(ValueError, "PICOVOICE_ACCESS_KEY"):
                    transcriber.transcribe_audio(audio.name)

    def test_missing_audio_file_is_rejected(self):
        with patch.dict(os.environ, {"PICOVOICE_ACCESS_KEY": "access-key"}, clear=True):
            transcriber = PicovoiceLeopardTranscription(
                temperature=0.3, leopard_factory=lambda **kwargs: FakeLeopard()
            )
            with self.assertRaisesRegex(ValueError, "does not exist"):
                transcriber.transcribe_audio("/tmp/does-not-exist.mp3")


if __name__ == "__main__":
    unittest.main()
