import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


class DeepInfraTranscriptionTest(unittest.TestCase):
    def test_transcribe_audio_posts_openai_compatible_payload(self):
        from sapat.transcription.deepinfra import DeepInfraTranscription

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as audio:
            audio.write(b"fake audio")
            audio_path = Path(audio.name)

        response = Mock()
        response.status_code = 200
        response.json.return_value = {"text": "transcribed with deepinfra"}

        env = {
            "DEEPINFRA_TOKEN": "test-token",
            "DEEPINFRA_MODEL": "openai/whisper-large-v3",
            "DEEPINFRA_API_ENDPOINT": "https://api.deepinfra.com/v1/audio/transcriptions",
        }

        try:
            with patch.dict(os.environ, env, clear=False), patch(
                "sapat.transcription.deepinfra.requests.post", return_value=response
            ) as post:
                result = DeepInfraTranscription(temperature=0.2).transcribe_audio(
                    str(audio_path),
                    language="en",
                    prompt="product names: Sapat and Daytona",
                    temperature=0.1,
                )

            self.assertEqual(result, {"text": "transcribed with deepinfra"})
            post.assert_called_once()

            _, kwargs = post.call_args
            self.assertEqual(
                kwargs["headers"], {"Authorization": "Bearer test-token"}
            )
            self.assertEqual(kwargs["data"]["model"], "openai/whisper-large-v3")
            self.assertEqual(kwargs["data"]["response_format"], "json")
            self.assertEqual(kwargs["data"]["language"], "en")
            self.assertEqual(kwargs["data"]["prompt"], "product names: Sapat and Daytona")
            self.assertEqual(kwargs["data"]["temperature"], 0.1)
            self.assertIn("file", kwargs["files"])
        finally:
            audio_path.unlink(missing_ok=True)

    def test_generate_corrected_transcript_uses_deepinfra_chat_endpoint(self):
        from sapat.transcription.deepinfra import DeepInfraTranscription

        choice = Mock()
        choice.message.content = "corrected transcript"
        completion = Mock()
        completion.choices = [choice]

        chat_completions = Mock()
        chat_completions.create.return_value = completion

        chat = Mock()
        chat.completions = chat_completions

        client = Mock()
        client.chat = chat

        env = {
            "DEEPINFRA_TOKEN": "test-token",
            "DEEPINFRA_MODEL_NAME_CHAT": "meta-llama/test-chat-model",
            "DEEPINFRA_OPENAI_BASE_URL": "https://api.deepinfra.com/v1/openai",
        }

        with patch.dict(os.environ, env, clear=False), patch(
            "sapat.transcription.deepinfra.OpenAI", return_value=client
        ) as openai, patch.object(
            DeepInfraTranscription,
            "transcribe_audio",
            return_value={"text": "raw transcript"},
        ):
            transcriber = DeepInfraTranscription(temperature=0.2)
            result = transcriber.generate_corrected_transcript(
                "audio.mp3",
                0.7,
                "Correct spelling only.",
            )

        self.assertEqual(result, "corrected transcript")
        openai.assert_called_once_with(
            api_key="test-token",
            base_url="https://api.deepinfra.com/v1/openai",
        )
        chat_completions.create.assert_called_once()

        _, kwargs = chat_completions.create.call_args
        self.assertEqual(kwargs["model"], "meta-llama/test-chat-model")
        self.assertEqual(kwargs["temperature"], 0.7)
        self.assertEqual(
            kwargs["messages"],
            [
                {
                    "role": "system",
                    "content": "Correct spelling only.",
                },
                {
                    "role": "user",
                    "content": "raw transcript",
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
