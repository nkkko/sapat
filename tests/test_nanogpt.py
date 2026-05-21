import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from click.testing import CliRunner

from sapat.script import main
from sapat.transcription.nanogpt import NanoGPTTranscription


class NanoGPTTranscriptionTest(unittest.TestCase):
    def _audio_file(self):
        handle = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        handle.write(b"audio")
        handle.close()
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return handle.name

    @patch.dict(
        os.environ,
        {
            "NANOGPT_API_KEY": "test-key",
            "NANOGPT_MODEL": "Whisper-Large-V3",
            "NANOGPT_API_ENDPOINT": "https://nano-gpt.com/api/v1/audio/transcriptions",
        },
        clear=True,
    )
    @patch("sapat.transcription.nanogpt.requests.post")
    def test_transcribe_audio_posts_openai_compatible_request(self, post):
        post.return_value = Mock(
            status_code=200,
            json=Mock(return_value={"text": "hello"}),
            text='{"text":"hello"}',
        )
        audio_file = self._audio_file()

        result = NanoGPTTranscription(temperature=0.2).transcribe_audio(
            audio_file,
            language="en",
            prompt="Product names: Daytona and Sapat",
        )

        self.assertEqual(result, {"text": "hello"})
        _, kwargs = post.call_args
        self.assertEqual(
            kwargs["headers"],
            {"Authorization": "Bearer test-key"},
        )
        self.assertEqual(
            kwargs["data"],
            {
                "model": "Whisper-Large-V3",
                "response_format": "json",
                "temperature": 0.2,
                "language": "en",
                "prompt": "Product names: Daytona and Sapat",
            },
        )
        self.assertIn("file", kwargs["files"])

    @patch.dict(os.environ, {}, clear=True)
    def test_transcribe_audio_requires_api_key(self):
        audio_file = self._audio_file()

        with self.assertRaisesRegex(ValueError, "NANOGPT_API_KEY"):
            NanoGPTTranscription(temperature=0.2).transcribe_audio(audio_file)

    @patch.dict(
        os.environ,
        {
            "NANOGPT_API_KEY": "test-key",
            "NANOGPT_MODEL": "Whisper-Large-V3",
            "NANOGPT_API_ENDPOINT": "https://nano-gpt.com/api/v1/audio/transcriptions",
            "NANOGPT_CHAT_MODEL": "openai/gpt-4o-mini",
            "NANOGPT_CHAT_ENDPOINT": "https://nano-gpt.com/api/v1/chat/completions",
        },
        clear=True,
    )
    @patch("sapat.transcription.nanogpt.requests.post")
    def test_generate_corrected_transcript_posts_chat_request(self, post):
        transcription_response = Mock(
            status_code=200,
            json=Mock(return_value={"text": "raw transkript"}),
            text='{"text":"raw transkript"}',
        )
        correction_response = Mock(
            status_code=200,
            json=Mock(
                return_value={
                    "choices": [
                        {"message": {"content": "Raw transcript."}}
                    ]
                }
            ),
            text='{"choices":[{"message":{"content":"Raw transcript."}}]}',
        )
        post.side_effect = [transcription_response, correction_response]
        audio_file = self._audio_file()

        result = NanoGPTTranscription(temperature=0.2).generate_corrected_transcript(
            audio_file,
            0.7,
            "Fix spelling only.",
        )

        self.assertEqual(result, "Raw transcript.")
        _, correction_kwargs = post.call_args_list[1]
        self.assertEqual(
            correction_kwargs["headers"],
            {
                "Authorization": "Bearer test-key",
                "Content-Type": "application/json",
            },
        )
        self.assertEqual(
            correction_kwargs["json"],
            {
                "model": "openai/gpt-4o-mini",
                "temperature": 0.7,
                "messages": [
                    {"role": "system", "content": "Fix spelling only."},
                    {"role": "user", "content": "raw transkript"},
                ],
            },
        )

    @patch.dict(
        os.environ,
        {
            "NANOGPT_API_KEY": "test-key",
            "NANOGPT_MODEL": "Whisper-Large-V3",
            "NANOGPT_API_ENDPOINT": "https://nano-gpt.com/api/v1/audio/transcriptions",
        },
        clear=True,
    )
    def test_generate_corrected_transcript_requires_chat_model(self):
        audio_file = self._audio_file()

        with self.assertRaisesRegex(ValueError, "NANOGPT_CHAT_MODEL"):
            NanoGPTTranscription(temperature=0.2).generate_corrected_transcript(
                audio_file,
                0.7,
                "Fix spelling only.",
            )

    def test_cli_accepts_nanogpt_provider(self):
        runner = CliRunner()

        with runner.isolated_filesystem():
            Path("sample.mp4").write_text("video")

            with patch("sapat.script.NanoGPTTranscription") as transcriber:
                result = runner.invoke(
                    main,
                    [
                        "sample.mp4",
                        "--api",
                        "nanogpt",
                        "--language",
                        "en",
                        "--prompt",
                        "Use project names",
                        "--temperature",
                        "0.1",
                    ],
                )

        self.assertEqual(result.exit_code, 0)
        transcriber.assert_called_once_with(temperature=0.1)
        transcriber.return_value.process_file.assert_called_once()
