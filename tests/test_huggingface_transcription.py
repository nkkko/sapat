import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from sapat.transcription.huggingface import HuggingFaceTranscription


class HuggingFaceTranscriptionTest(unittest.TestCase):
    def test_transcribe_audio_posts_mp3_bytes_to_model_endpoint(self):
        with tempfile.NamedTemporaryFile(suffix=".mp3") as audio:
            audio.write(b"audio bytes")
            audio.flush()

            response = Mock()
            response.status_code = 200
            response.json.return_value = {"text": "hello from hugging face"}
            response.text = '{"text":"hello from hugging face"}'

            with patch.dict(
                os.environ,
                {
                    "HUGGINGFACE_API_KEY": "hf_test",
                    "HUGGINGFACE_MODEL": "openai/whisper-large-v3-turbo",
                },
            ), patch("sapat.transcription.huggingface.requests.post", return_value=response) as post:
                transcriber = HuggingFaceTranscription(temperature=0.3)
                result = transcriber.transcribe_audio(audio.name)

        self.assertEqual(result, {"text": "hello from hugging face"})
        post.assert_called_once()
        url = post.call_args.args[0]
        self.assertEqual(url, "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo")
        self.assertEqual(post.call_args.kwargs["data"], b"audio bytes")
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer hf_test")
        self.assertEqual(post.call_args.kwargs["headers"]["Content-Type"], "audio/mpeg")

    def test_transcribe_audio_uses_endpoint_override(self):
        with tempfile.NamedTemporaryFile(suffix=".mp3") as audio:
            audio.write(b"audio bytes")
            audio.flush()

            response = Mock()
            response.status_code = 200
            response.json.return_value = {"text": "endpoint output"}
            response.text = '{"text":"endpoint output"}'

            with patch.dict(
                os.environ,
                {
                    "HUGGINGFACE_API_KEY": "hf_test",
                    "HUGGINGFACE_API_ENDPOINT": "https://example.endpoints.huggingface.cloud",
                },
            ), patch("sapat.transcription.huggingface.requests.post", return_value=response) as post:
                transcriber = HuggingFaceTranscription(temperature=0.3)
                result = transcriber.transcribe_audio(audio.name)

        self.assertEqual(result, {"text": "endpoint output"})
        self.assertEqual(post.call_args.args[0], "https://example.endpoints.huggingface.cloud")

    def test_transcribe_audio_sets_wav_content_type(self):
        with tempfile.NamedTemporaryFile(suffix=".wav") as audio:
            audio.write(b"audio bytes")
            audio.flush()

            response = Mock()
            response.status_code = 200
            response.json.return_value = {"text": "wav output"}
            response.text = '{"text":"wav output"}'

            with patch.dict(os.environ, {"HUGGINGFACE_API_KEY": "hf_test"}), patch(
                "sapat.transcription.huggingface.requests.post", return_value=response
            ) as post:
                transcriber = HuggingFaceTranscription(temperature=0.3)
                result = transcriber.transcribe_audio(audio.name)

        self.assertEqual(result, {"text": "wav output"})
        self.assertEqual(post.call_args.kwargs["headers"]["Content-Type"], "audio/wav")

    def test_transcribe_audio_requires_api_key(self):
        with tempfile.NamedTemporaryFile(suffix=".mp3") as audio:
            audio.write(b"audio bytes")
            audio.flush()

            with patch.dict(os.environ, {"HUGGINGFACE_API_KEY": ""}):
                transcriber = HuggingFaceTranscription(temperature=0.3)

                with self.assertRaisesRegex(ValueError, "HUGGINGFACE_API_KEY"):
                    transcriber.transcribe_audio(audio.name)


if __name__ == "__main__":
    unittest.main()
