import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from sapat.script import main
from sapat.transcription.speechmatics import SpeechmaticsTranscription


class SpeechmaticsTranscriptionTest(unittest.TestCase):
    def setUp(self):
        self.previous_key = os.environ.get("SPEECHMATICS_API_KEY")
        os.environ["SPEECHMATICS_API_KEY"] = "test-key"

    def tearDown(self):
        if self.previous_key is None:
            os.environ.pop("SPEECHMATICS_API_KEY", None)
        else:
            os.environ["SPEECHMATICS_API_KEY"] = self.previous_key

    @patch("sapat.transcription.speechmatics.requests.get")
    @patch("sapat.transcription.speechmatics.requests.post")
    def test_transcribe_audio_creates_job_and_fetches_text_transcript(self, mock_post, mock_get):
        create_response = MagicMock(status_code=201)
        create_response.json.return_value = {"id": "job-123"}

        status_response = MagicMock(status_code=200)
        status_response.json.return_value = {"job": {"status": "done"}}

        transcript_response = MagicMock(status_code=200, text="hello from speechmatics")

        mock_post.return_value = create_response
        mock_get.side_effect = [status_response, transcript_response]

        with tempfile.NamedTemporaryFile(suffix=".mp3") as audio:
            audio.write(b"audio")
            audio.flush()

            transcriber = SpeechmaticsTranscription(
                temperature=0.3,
                poll_interval_seconds=0,
                max_poll_seconds=1,
            )
            result = transcriber.transcribe_audio(audio.name, language="en")

        self.assertEqual(result, "hello from speechmatics")
        posted_config = json.loads(mock_post.call_args.kwargs["data"]["config"])
        self.assertEqual(posted_config["type"], "transcription")
        self.assertEqual(posted_config["transcription_config"]["language"], "en")
        self.assertEqual(
            mock_get.call_args_list[1].kwargs["params"],
            {"format": "txt"},
        )

    @patch("sapat.script.SpeechmaticsTranscription")
    def test_cli_routes_speechmatics_api(self, mock_transcriber):
        runner = CliRunner()

        with runner.isolated_filesystem():
            Path("sample.mp4").write_bytes(b"video")
            result = runner.invoke(main, ["sample.mp4", "--api", "speechmatics"])

        self.assertEqual(result.exit_code, 0)
        mock_transcriber.assert_called_once_with(temperature=0.3)
        mock_transcriber.return_value.process_file.assert_called_once()


if __name__ == "__main__":
    unittest.main()
