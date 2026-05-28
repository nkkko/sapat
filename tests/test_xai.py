import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from click.testing import CliRunner


class XAITranscriptionTests(unittest.TestCase):
    def _audio_file(self, suffix=".mp3"):
        handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        handle.write(b"audio")
        handle.close()
        self.addCleanup(lambda: os.path.exists(handle.name) and os.unlink(handle.name))
        return handle.name

    def test_transcribe_audio_posts_to_xai_stt_with_file_last(self):
        from sapat.transcription.xai import XAITranscription

        audio_file = self._audio_file()
        response = Mock(status_code=200)
        response.json.return_value = {"text": "hello", "duration": 1.25, "words": []}

        env = {
            "XAI_API_KEY": "test-key",
            "XAI_STT_KEYTERMS": "Daytona,Sapat",
            "XAI_STT_DIARIZE": "true",
            "XAI_STT_MULTICHANNEL": "false",
        }

        with patch.dict(os.environ, env, clear=True), patch(
            "sapat.transcription.xai.requests.post", return_value=response
        ) as post:
            result = XAITranscription(temperature=0.3).transcribe_audio(
                audio_file, language="en"
            )

        self.assertEqual(result["text"], "hello")
        post.assert_called_once()
        _, kwargs = post.call_args
        self.assertEqual(post.call_args.args[0], "https://api.x.ai/v1/stt")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(
            kwargs["data"],
            [
                ("format", "true"),
                ("language", "en"),
                ("diarize", "true"),
                ("multichannel", "false"),
                ("keyterm", "Daytona"),
                ("keyterm", "Sapat"),
            ],
        )
        self.assertEqual(list(kwargs["files"].keys()), ["file"])
        self.assertEqual(kwargs["files"]["file"][0], os.path.basename(audio_file))

    def test_transcribe_audio_requires_api_key(self):
        from sapat.transcription.xai import XAITranscription

        audio_file = self._audio_file()

        with patch.dict(os.environ, {"XAI_API_KEY": ""}, clear=True):
            transcriber = XAITranscription(temperature=0.3)
            with self.assertRaisesRegex(ValueError, "XAI_API_KEY"):
                transcriber.transcribe_audio(audio_file, language="en")

    def test_transcribe_audio_rejects_oversized_files(self):
        from sapat.transcription.xai import XAITranscription

        audio_file = self._audio_file()

        with patch.dict(os.environ, {"XAI_API_KEY": "test-key"}, clear=True), patch(
            "sapat.transcription.xai.os.path.getsize",
            return_value=(501 * 1024 * 1024),
        ):
            transcriber = XAITranscription(temperature=0.3)
            with self.assertRaisesRegex(Exception, "500 MB"):
                transcriber.transcribe_audio(audio_file, language="en")

    def test_cli_accepts_xai_provider(self):
        from sapat import script

        runner = CliRunner()

        with runner.isolated_filesystem():
            with open("clip.mp4", "wb") as f:
                f.write(b"video")

            with patch.object(script, "XAITranscription") as transcriber_cls:
                transcriber = transcriber_cls.return_value
                result = runner.invoke(script.main, ["clip.mp4", "--api", "xai"])

        self.assertEqual(result.exit_code, 0)
        transcriber_cls.assert_called_once_with(temperature=0.3)
        transcriber.process_file.assert_called_once()


if __name__ == "__main__":
    unittest.main()
