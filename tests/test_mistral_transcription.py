import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from sapat.transcription.mistral import MistralTranscription


class MistralTranscriptionTest(unittest.TestCase):
    def make_audio_file(self):
        audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        audio.write(b"fake audio")
        audio.close()
        self.addCleanup(lambda: os.path.exists(audio.name) and os.unlink(audio.name))
        return audio.name

    @patch.dict(
        os.environ,
        {
            "MISTRAL_API_KEY": "test-key",
            "MISTRAL_MODEL": "voxtral-mini-latest",
            "MISTRAL_API_ENDPOINT": "https://api.mistral.ai/v1/audio/transcriptions",
            "MISTRAL_MODEL_NAME_CHAT": "mistral-small-latest",
            "MISTRAL_CHAT_API_ENDPOINT": "https://api.mistral.ai/v1/chat/completions",
        },
    )
    @patch("sapat.transcription.mistral.requests.post")
    def test_transcribe_audio_sends_mistral_request(self, post):
        response = MagicMock(status_code=200)
        response.json.return_value = {"text": "Hello from Mistral"}
        post.return_value = response
        transcriber = MistralTranscription(temperature=0.2)

        result = transcriber.transcribe_audio(
            self.make_audio_file(),
            language="en",
            prompt="Product names: Daytona, Sapat",
        )

        self.assertEqual(result, {"text": "Hello from Mistral"})
        _, kwargs = post.call_args
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(kwargs["data"]["model"], "voxtral-mini-latest")
        self.assertEqual(kwargs["data"]["language"], "en")
        self.assertEqual(kwargs["data"]["context_bias"], "Product names: Daytona, Sapat")
        self.assertIn("file", kwargs["files"])

    @patch.dict(
        os.environ,
        {
            "MISTRAL_API_KEY": "test-key",
            "MISTRAL_MODEL": "voxtral-mini-latest",
            "MISTRAL_API_ENDPOINT": "https://api.mistral.ai/v1/audio/transcriptions",
            "MISTRAL_MODEL_NAME_CHAT": "mistral-small-latest",
            "MISTRAL_CHAT_API_ENDPOINT": "https://api.mistral.ai/v1/chat/completions",
        },
    )
    @patch("sapat.transcription.mistral.requests.post")
    def test_generate_corrected_transcript_uses_chat_endpoint(self, post):
        transcription_response = MagicMock(status_code=200)
        transcription_response.json.return_value = {"text": "raw transcript"}
        chat_response = MagicMock(status_code=200)
        chat_response.json.return_value = {
            "choices": [{"message": {"content": "corrected transcript"}}]
        }
        post.side_effect = [transcription_response, chat_response]
        transcriber = MistralTranscription(temperature=0.2)

        result = transcriber.generate_corrected_transcript(
            self.make_audio_file(),
            0.1,
            "Correct spelling only.",
        )

        self.assertEqual(result, "corrected transcript")
        chat_call = post.call_args_list[1]
        self.assertEqual(chat_call.args[0], "https://api.mistral.ai/v1/chat/completions")
        self.assertEqual(chat_call.kwargs["json"]["model"], "mistral-small-latest")
        self.assertEqual(chat_call.kwargs["json"]["messages"][1]["content"], "raw transcript")


if __name__ == "__main__":
    unittest.main()
