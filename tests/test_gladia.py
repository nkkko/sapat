import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from click.testing import CliRunner

sys.modules.setdefault("groq", types.SimpleNamespace(Groq=object))
sys.modules.setdefault(
    "openai", types.SimpleNamespace(OpenAI=object, AzureOpenAI=object)
)

from sapat.script import main
from sapat.transcription.gladia import GladiaTranscription


class GladiaTranscriptionTests(unittest.TestCase):
    def test_transcribe_audio_uploads_creates_job_and_polls_result(self):
        with tempfile.NamedTemporaryFile(suffix=".mp3") as audio:
            audio.write(b"fake audio")
            audio.flush()

            responses = [
                Mock(
                    status_code=201,
                    json=Mock(return_value={"audio_url": "https://api.gladia.io/file/1"}),
                    text="",
                ),
                Mock(
                    status_code=201,
                    json=Mock(
                        return_value={
                            "id": "job-1",
                            "result_url": "https://api.gladia.io/v2/pre-recorded/job-1",
                        }
                    ),
                    text="",
                ),
            ]
            poll_response = Mock(
                status_code=200,
                json=Mock(
                    return_value={
                        "status": "done",
                        "result": {
                            "transcription": {
                                "full_transcript": "Transcript from Gladia."
                            }
                        },
                    }
                ),
                text="",
            )

            with patch.dict(os.environ, {"GLADIA_API_KEY": "test-key"}), patch(
                "sapat.transcription.gladia.requests.post", side_effect=responses
            ) as post, patch(
                "sapat.transcription.gladia.requests.get", return_value=poll_response
            ) as get:
                transcriber = GladiaTranscription(
                    temperature=0.3, poll_interval_seconds=0, timeout_seconds=1
                )

                result = transcriber.transcribe_audio(audio.name, language="en")

        self.assertEqual(result, {"text": "Transcript from Gladia."})
        self.assertEqual(post.call_count, 2)
        self.assertEqual(get.call_count, 1)
        self.assertEqual(post.call_args_list[1].kwargs["json"]["language_config"], {
            "languages": ["en"],
            "code_switching": False,
        })

    def test_cli_routes_gladia_api_choice(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as video:
            with patch("sapat.script.GladiaTranscription") as transcriber_cls:
                transcriber = transcriber_cls.return_value
                runner = CliRunner()
                result = runner.invoke(main, [video.name, "--api", "gladia"])

        self.assertEqual(result.exit_code, 0)
        transcriber_cls.assert_called_once_with(temperature=0.3)
        transcriber.process_file.assert_called_once()
        self.assertEqual(Path(transcriber.process_file.call_args.args[0]), Path(video.name))


if __name__ == "__main__":
    unittest.main()
