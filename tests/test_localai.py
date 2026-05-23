import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from sapat import script


class TestLocalAITranscription(unittest.TestCase):
    def test_cli_routes_localai_provider(self):
        calls = []

        class FakeLocalAITranscription:
            def __init__(self, temperature):
                self.temperature = temperature

            def process_file(self, input_file, language, prompt, temperature, quality, correct):
                calls.append(
                    {
                        "input_file": input_file,
                        "language": language,
                        "prompt": prompt,
                        "temperature": temperature,
                        "quality": quality,
                        "correct": correct,
                    }
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "demo.mp4"
            video_path.write_bytes(b"not a real video")

            with patch(
                "sapat.transcription.localai.LocalAITranscription",
                FakeLocalAITranscription,
            ):
                result = CliRunner().invoke(
                    script.main,
                    [
                        str(video_path),
                        "--api",
                        "localai",
                        "--language",
                        "en",
                        "--prompt",
                        "Product demo vocabulary",
                        "--temperature",
                        "0.2",
                        "--quality",
                        "H",
                    ],
                )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["input_file"].name, "demo.mp4")
        self.assertEqual(calls[0]["language"], "en")
        self.assertEqual(calls[0]["prompt"], "Product demo vocabulary")
        self.assertEqual(calls[0]["temperature"], 0.2)
        self.assertEqual(calls[0]["quality"], "H")
        self.assertFalse(calls[0]["correct"])

    def test_localai_posts_openai_compatible_transcription_request(self):
        from sapat.transcription.localai import LocalAITranscription

        captured = {}

        class FakeResponse:
            status_code = 200
            text = json.dumps({"text": "hello from localai"})

            def json(self):
                return {"text": "hello from localai"}

        def fake_post(endpoint, headers, data, files):
            captured["endpoint"] = endpoint
            captured["headers"] = headers
            captured["data"] = data
            captured["file_name"] = files["file"].name
            return FakeResponse()

        with tempfile.NamedTemporaryFile(suffix=".mp3") as audio_file:
            audio_file.write(b"audio")
            audio_file.flush()

            with patch.dict(
                os.environ,
                {
                    "LOCALAI_BASE_URL": "http://localhost:8080",
                    "LOCALAI_MODEL": "whisper-large-v3",
                    "LOCALAI_API_KEY": "test-token",
                },
                clear=False,
            ):
                transcriber = LocalAITranscription(temperature=0.4)
                with patch("sapat.transcription.localai.requests.post", fake_post):
                    result = transcriber.transcribe_audio(
                        audio_file.name,
                        language="en",
                        prompt="meeting vocabulary",
                        response_format="json",
                    )

        self.assertEqual(result, {"text": "hello from localai"})
        self.assertEqual(captured["endpoint"], "http://localhost:8080/v1/audio/transcriptions")
        self.assertEqual(captured["headers"], {"Authorization": "Bearer test-token"})
        self.assertEqual(captured["data"]["model"], "whisper-large-v3")
        self.assertEqual(captured["data"]["response_format"], "json")
        self.assertEqual(captured["data"]["temperature"], 0.4)
        self.assertEqual(captured["data"]["language"], "en")
        self.assertEqual(captured["data"]["prompt"], "meeting vocabulary")
        self.assertTrue(captured["file_name"].endswith(".mp3"))


if __name__ == "__main__":
    unittest.main()
