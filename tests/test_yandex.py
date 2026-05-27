import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from click.testing import CliRunner

from sapat.script import main
from sapat.transcription.yandex import YandexSpeechKitTranscription


class YandexSpeechKitTranscriptionTests(unittest.TestCase):
    def setUp(self):
        self.env = {
            "YANDEX_API_KEY": "test-api-key",
            "YANDEX_IAM_TOKEN": "",
            "YANDEX_FOLDER_ID": "folder-123",
            "YANDEX_API_ENDPOINT": "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize",
            "YANDEX_LANGUAGE": "en-US",
            "YANDEX_TOPIC": "general",
            "YANDEX_AUDIO_FORMAT": "oggopus",
        }

    def _write_prepared_file(self, command, check):
        Path(command[-1]).write_bytes(b"fake oggopus")

    def test_transcribe_audio_posts_binary_audio_and_options(self):
        response = Mock(status_code=200)
        response.json.return_value = {"result": "hello world"}

        with tempfile.NamedTemporaryFile(suffix=".mp3") as audio_file, patch.dict(os.environ, self.env, clear=False), patch(
            "sapat.transcription.yandex.subprocess.run", side_effect=self._write_prepared_file
        ) as ffmpeg, patch("sapat.transcription.yandex.requests.post", return_value=response) as post:
            audio_file.write(b"fake audio")
            audio_file.flush()

            result = YandexSpeechKitTranscription(temperature=0.3).transcribe_audio(audio_file.name, language="en")

        self.assertEqual(result, {"text": "hello world", "result": "hello world"})
        ffmpeg.assert_called_once()
        self.assertFalse(Path(ffmpeg.call_args.args[0][-1]).exists())
        self.assertEqual(post.call_args.args[0], "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize")
        self.assertEqual(
            post.call_args.kwargs["headers"],
            {
                "Authorization": "Api-Key test-api-key",
                "Content-Type": "application/octet-stream",
            },
        )
        self.assertEqual(
            post.call_args.kwargs["params"],
            {
                "lang": "en-US",
                "topic": "general",
                "format": "oggopus",
            },
        )

    def test_transcribe_audio_supports_iam_token_auth_with_folder_id(self):
        response = Mock(status_code=200)
        response.json.return_value = {"result": "token auth"}
        env = {
            "YANDEX_API_KEY": "",
            "YANDEX_IAM_TOKEN": "iam-token",
            "YANDEX_FOLDER_ID": "folder-123",
        }

        with tempfile.NamedTemporaryFile(suffix=".mp3") as audio_file, patch.dict(os.environ, env, clear=False), patch(
            "sapat.transcription.yandex.subprocess.run", side_effect=self._write_prepared_file
        ), patch("sapat.transcription.yandex.requests.post", return_value=response) as post:
            audio_file.write(b"fake audio")
            audio_file.flush()

            result = YandexSpeechKitTranscription(temperature=0.3).transcribe_audio(audio_file.name)

        self.assertEqual(result["text"], "token auth")
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer iam-token")
        self.assertEqual(post.call_args.kwargs["params"]["folderId"], "folder-123")

    def test_transcribe_audio_requires_auth(self):
        with tempfile.NamedTemporaryFile(suffix=".mp3") as audio_file, patch.dict(
            os.environ, {"YANDEX_API_KEY": "", "YANDEX_IAM_TOKEN": ""}, clear=False
        ):
            audio_file.write(b"fake audio")
            audio_file.flush()

            with self.assertRaises(ValueError):
                YandexSpeechKitTranscription(temperature=0.3).transcribe_audio(audio_file.name)

    def test_transcribe_audio_raises_api_errors(self):
        response = Mock(status_code=401, text="unauthorized")

        with tempfile.NamedTemporaryFile(suffix=".mp3") as audio_file, patch.dict(os.environ, self.env, clear=False), patch(
            "sapat.transcription.yandex.subprocess.run", side_effect=self._write_prepared_file
        ), patch("sapat.transcription.yandex.requests.post", return_value=response):
            audio_file.write(b"fake audio")
            audio_file.flush()

            with self.assertRaisesRegex(Exception, "Yandex SpeechKit transcription failed"):
                YandexSpeechKitTranscription(temperature=0.3).transcribe_audio(audio_file.name)

    def test_cli_routes_to_yandex_transcriber(self):
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".mp4") as video_file, patch(
            "sapat.script.YandexSpeechKitTranscription"
        ) as transcriber:
            result = runner.invoke(main, [video_file.name, "--api", "yandex"])

        self.assertEqual(result.exit_code, 0)
        transcriber.assert_called_once_with(temperature=0.3)
        transcriber.return_value.process_file.assert_called_once()


if __name__ == "__main__":
    unittest.main()
