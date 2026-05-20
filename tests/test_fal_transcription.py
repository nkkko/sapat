import os
import tempfile
import unittest
from unittest.mock import patch

from sapat.transcription.fal import FalTranscription


class FalTranscriptionTests(unittest.TestCase):
    def test_transcribe_audio_uploads_file_and_subscribes_to_whisper(self):
        with tempfile.NamedTemporaryFile(suffix=".mp3") as audio_file:
            audio_file.write(b"audio")
            audio_file.flush()

            with patch.dict(os.environ, {"FAL_KEY": "test-key"}):
                with patch("sapat.transcription.fal.fal_client.upload_file", return_value="https://fal.media/test.mp3") as upload:
                    with patch("sapat.transcription.fal.fal_client.subscribe", return_value={"text": "hello world"}) as subscribe:
                        transcriber = FalTranscription(temperature=0.3)
                        result = transcriber.transcribe_audio(
                            audio_file.name,
                            language="en",
                            prompt="Daytona, Sapat",
                        )

        self.assertEqual(result, {"text": "hello world"})
        upload.assert_called_once_with(audio_file.name)
        subscribe.assert_called_once_with(
            "fal-ai/whisper",
            arguments={
                "audio_url": "https://fal.media/test.mp3",
                "task": "transcribe",
                "chunk_level": "segment",
                "batch_size": 64,
                "language": "en",
                "prompt": "Daytona, Sapat",
            },
        )

    def test_transcribe_audio_requires_api_key(self):
        with tempfile.NamedTemporaryFile(suffix=".mp3") as audio_file:
            audio_file.write(b"audio")
            audio_file.flush()

            with patch.dict(os.environ, {"FAL_KEY": ""}):
                transcriber = FalTranscription(temperature=0.3)
                with self.assertRaisesRegex(ValueError, "FAL_KEY"):
                    transcriber.transcribe_audio(audio_file.name)


if __name__ == "__main__":
    unittest.main()
