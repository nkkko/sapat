import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from sapat.transcription.deepgram import DeepgramTranscription


class DeepgramTranscriptionTest(unittest.TestCase):
    def test_extracts_transcript_from_deepgram_payload(self):
        transcriber = DeepgramTranscription(temperature=0.3)

        transcript = transcriber._extract_transcript({
            "results": {
                "channels": [
                    {
                        "alternatives": [
                            {
                                "transcript": "hello from deepgram"
                            }
                        ]
                    }
                ]
            }
        })

        self.assertEqual(transcript, "hello from deepgram")

    def test_uses_expected_content_types(self):
        transcriber = DeepgramTranscription(temperature=0.3)

        self.assertEqual(transcriber._content_type_for("clip.mp3"), "audio/mpeg")
        self.assertEqual(transcriber._content_type_for("clip.wav"), "audio/wav")
        self.assertEqual(transcriber._content_type_for("clip.flac"), "audio/flac")

    @patch.dict(os.environ, {"DEEPGRAM_API_KEY": "test-key"}, clear=True)
    @patch("sapat.transcription.deepgram.requests.post")
    def test_posts_audio_to_deepgram(self, post):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "results": {
                "channels": [
                    {
                        "alternatives": [
                            {
                                "transcript": "transcribed text"
                            }
                        ]
                    }
                ]
            }
        }
        post.return_value = response

        transcriber = DeepgramTranscription(temperature=0.4)

        audio_path = self._write_temp_audio_file()
        try:
            result = transcriber.transcribe_audio(str(audio_path), language="en")
        finally:
            audio_path.unlink(missing_ok=True)


        self.assertEqual(result["text"], "transcribed text")
        _, kwargs = post.call_args
        self.assertEqual(kwargs["headers"]["Authorization"], "Token test-key")
        self.assertEqual(kwargs["headers"]["Content-Type"], "audio/mpeg")
        self.assertEqual(kwargs["params"]["model"], "nova-3")
        self.assertEqual(kwargs["params"]["language"], "en")

    @patch.dict(os.environ, {}, clear=True)
    def test_requires_deepgram_api_key(self):
        transcriber = DeepgramTranscription(temperature=0.3)

        audio_path = self._write_temp_audio_file()
        try:
            with self.assertRaisesRegex(ValueError, "DEEPGRAM_API_KEY"):
                transcriber.transcribe_audio(str(audio_path))
        finally:
            audio_path.unlink(missing_ok=True)

    def _write_temp_audio_file(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as audio_file:
            audio_file.write(b"fake audio")
            return Path(audio_file.name)


if __name__ == "__main__":
    unittest.main()
