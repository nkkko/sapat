import ast
import base64
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import requests


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sapat.transcription.gemini import GeminiTranscription


def ast_string_value(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Str):
        return node.s
    return None


class FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class GeminiTranscriptionTests(unittest.TestCase):
    def _temporary_audio_file(self, suffix=".mp3", content=b"audio bytes"):
        temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        try:
            temp_file.write(content)
            return temp_file.name
        finally:
            temp_file.close()
            self.addCleanup(self._remove_file, temp_file.name)

    def _remove_file(self, path):
        if os.path.exists(path):
            os.remove(path)

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True)
    @patch("sapat.transcription.gemini.requests.post")
    def test_request_construction_uses_inline_base64_audio(self, mock_post):
        audio_content = b"example audio"
        audio_file = self._temporary_audio_file(".mp3", audio_content)
        mock_post.return_value = FakeResponse(
            200,
            {
                "candidates": [
                    {"content": {"parts": [{"text": "hola mundo"}]}}
                ]
            },
        )

        result = GeminiTranscription(temperature=0.2).transcribe_audio(
            audio_file,
            language="es",
            prompt="Speaker says Sapat.",
        )

        self.assertEqual(result, "hola mundo")
        mock_post.assert_called_once()
        request_url = mock_post.call_args[0][0]
        request_kwargs = mock_post.call_args[1]
        request_json = request_kwargs["json"]
        parts = request_json["contents"][0]["parts"]

        self.assertEqual(
            request_url,
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-2.0-flash:generateContent",
        )
        self.assertEqual(request_kwargs["headers"]["x-goog-api-key"], "test-key")
        self.assertEqual(request_kwargs["headers"]["Content-Type"], "application/json")
        self.assertEqual(request_json["generationConfig"]["temperature"], 0.2)
        self.assertIn("Language: es.", parts[0]["text"])
        self.assertIn("Speaker says Sapat.", parts[0]["text"])
        self.assertEqual(parts[1]["inline_data"]["mime_type"], "audio/mp3")
        self.assertEqual(
            parts[1]["inline_data"]["data"],
            base64.b64encode(audio_content).decode("ascii"),
        )

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True)
    @patch("sapat.transcription.gemini.requests.post")
    def test_response_parsing_concatenates_text_parts(self, mock_post):
        audio_file = self._temporary_audio_file(".wav")
        mock_post.return_value = FakeResponse(
            200,
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": " first part "},
                                {"inline_data": {"mime_type": "audio/wav"}},
                                {"text": "second part"},
                            ]
                        }
                    }
                ]
            },
        )

        result = GeminiTranscription(temperature=0.3).transcribe_audio(audio_file)

        self.assertEqual(result, "first part\nsecond part")

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "GEMINI_API_KEY"):
            GeminiTranscription(temperature=0.3)

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True)
    @patch("sapat.transcription.gemini.requests.post")
    def test_validation_rejects_missing_unsupported_and_large_files(self, mock_post):
        transcriber = GeminiTranscription(temperature=0.3)

        with self.assertRaises(FileNotFoundError):
            transcriber.transcribe_audio("/tmp/does-not-exist.mp3")

        unsupported_file = self._temporary_audio_file(".ogg")
        with self.assertRaisesRegex(ValueError, "Unsupported audio file format"):
            transcriber.transcribe_audio(unsupported_file)

        large_file = self._temporary_audio_file(".mp3", b"")
        with open(large_file, "wb") as handle:
            handle.seek(14 * 1024 * 1024)
            handle.write(b"x")
        with self.assertRaisesRegex(ValueError, "Gemini inline audio limit"):
            transcriber.transcribe_audio(large_file)

        mock_post.assert_not_called()

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True)
    @patch("sapat.transcription.gemini.requests.post")
    def test_api_error_raises_helpful_exception(self, mock_post):
        audio_file = self._temporary_audio_file(".flac")
        mock_post.return_value = FakeResponse(
            400,
            {"error": {"message": "API key not valid"}},
            text='{"error": {"message": "API key not valid"}}',
        )

        with self.assertRaisesRegex(RuntimeError, "API key not valid"):
            GeminiTranscription(temperature=0.3).transcribe_audio(audio_file)

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True)
    @patch("sapat.transcription.gemini.requests.post")
    def test_empty_response_raises_helpful_exception(self, mock_post):
        audio_file = self._temporary_audio_file(".mp3")
        mock_post.return_value = FakeResponse(
            200,
            {
                "candidates": [
                    {"content": {"parts": []}, "finishReason": "STOP"}
                ]
            },
        )

        with self.assertRaisesRegex(RuntimeError, "empty response"):
            GeminiTranscription(temperature=0.3).transcribe_audio(audio_file)

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True)
    @patch("sapat.transcription.gemini.requests.post")
    def test_request_exception_is_wrapped(self, mock_post):
        audio_file = self._temporary_audio_file(".mp3")
        mock_post.side_effect = requests.RequestException("timeout")

        with self.assertRaisesRegex(RuntimeError, "request failed: timeout"):
            GeminiTranscription(temperature=0.3).transcribe_audio(audio_file)

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True)
    @patch("sapat.transcription.gemini.requests.post")
    def test_non_json_response_is_rejected(self, mock_post):
        audio_file = self._temporary_audio_file(".mp3")
        mock_post.return_value = FakeResponse(200, ValueError("not json"), text="oops")

        with self.assertRaisesRegex(RuntimeError, "non-JSON response"):
            GeminiTranscription(temperature=0.3).transcribe_audio(audio_file)

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True)
    @patch("sapat.transcription.gemini.requests.post")
    def test_prompt_feedback_error_is_reported(self, mock_post):
        audio_file = self._temporary_audio_file(".mp3")
        mock_post.return_value = FakeResponse(
            200,
            {"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}},
        )

        with self.assertRaisesRegex(RuntimeError, "Prompt feedback"):
            GeminiTranscription(temperature=0.3).transcribe_audio(audio_file)

    @patch.dict(
        os.environ,
        {
            "GEMINI_API_KEY": "test-key",
            "GEMINI_MODEL": "gemini-1.5-flash",
            "GEMINI_API_ENDPOINT_TEMPLATE": "https://example.test/{model}",
        },
        clear=True,
    )
    @patch("sapat.transcription.gemini.requests.post")
    def test_custom_model_and_endpoint_template_are_used(self, mock_post):
        audio_file = self._temporary_audio_file(".mp3")
        mock_post.return_value = FakeResponse(
            200,
            {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]},
        )

        GeminiTranscription(temperature=0.3).transcribe_audio(audio_file)

        self.assertEqual(mock_post.call_args[0][0], "https://example.test/gemini-1.5-flash")

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True)
    @patch("sapat.transcription.gemini.GeminiTranscription.transcribe_audio")
    def test_generate_corrected_transcript_uses_gemini_prompt(self, mock_transcribe):
        mock_transcribe.return_value = "clean transcript"
        result = GeminiTranscription(temperature=0.3).generate_corrected_transcript(
            "clip.mp3", 0.7, "Keep product names correct."
        )

        self.assertEqual(result, "clean transcript")
        _, kwargs = mock_transcribe.call_args
        self.assertEqual(kwargs["temperature"], 0.7)
        self.assertIn("light cleanup", kwargs["prompt"])
        self.assertIn("Keep product names correct.", kwargs["prompt"])


class GeminiCliTests(unittest.TestCase):
    def test_cli_choice_includes_gemini(self):
        script_path = ROOT / "src" / "sapat" / "script.py"
        source = script_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        main_function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )

        api_choices = None
        for decorator in main_function.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            if not getattr(decorator.func, "attr", None) == "option":
                continue
            option_names = [ast_string_value(arg) for arg in decorator.args]
            if "--api" not in option_names:
                continue
            for keyword in decorator.keywords:
                if keyword.arg == "type" and isinstance(keyword.value, ast.Call):
                    choice_args = keyword.value.args
                    if choice_args and isinstance(
                        choice_args[0], (ast.List, ast.Tuple)
                    ):
                        api_choices = [
                            ast_string_value(item) for item in choice_args[0].elts
                        ]

        self.assertIsNotNone(api_choices)
        self.assertIn("gemini", api_choices)
        self.assertIn('elif api.lower() == "gemini":', source)
        self.assertIn("GeminiTranscription(temperature=temperature)", source)


if __name__ == "__main__":
    unittest.main()
