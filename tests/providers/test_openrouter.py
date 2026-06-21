# ABOUTME: Tests for the OpenRouter speech-to-text provider
# ABOUTME: Verifies JSON/base64 request shape, aliases, availability, and errors

import base64
import os
from unittest.mock import patch

import pytest

from sapat.providers.base import TranscriptionResult


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self.payload = payload or {}
        self.text = text

    def json(self):
        return self.payload


class TestOpenRouterProvider:
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False)
    @patch("sapat.providers.openrouter.requests.post")
    def test_transcribe_sends_base64_json_request(self, mock_post, tmp_path):
        audio_file = tmp_path / "sample.wav"
        audio_file.write_bytes(b"fake wav data")
        mock_post.return_value = FakeResponse(
            payload={
                "text": "hello openrouter",
                "usage": {"seconds": 1.5, "cost": 0.0001},
            }
        )

        from sapat.providers.openrouter import OpenRouterProvider

        provider = OpenRouterProvider()
        result = provider.transcribe(
            str(audio_file),
            model="openai/whisper-large-v3",
            language="en",
            temperature=0.2,
        )

        assert isinstance(result, TranscriptionResult)
        assert result.text == "hello openrouter"
        assert result.duration == 1.5
        assert result.raw_response["usage"]["cost"] == 0.0001

        _, kwargs = mock_post.call_args
        assert mock_post.call_args.args[0] == (
            "https://openrouter.ai/api/v1/audio/transcriptions"
        )
        assert kwargs["headers"]["Authorization"] == "Bearer test-key"
        assert kwargs["headers"]["Content-Type"] == "application/json"
        assert kwargs["json"]["model"] == "openai/whisper-large-v3"
        assert kwargs["json"]["language"] == "en"
        assert kwargs["json"]["temperature"] == 0.2
        assert kwargs["json"]["input_audio"]["format"] == "wav"
        assert kwargs["json"]["input_audio"]["data"] == base64.b64encode(
            b"fake wav data"
        ).decode("ascii")

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False)
    @patch("sapat.providers.openrouter.requests.post")
    def test_auto_language_omits_language_field(self, mock_post, tmp_path):
        audio_file = tmp_path / "sample.mp3"
        audio_file.write_bytes(b"fake mp3 data")
        mock_post.return_value = FakeResponse(payload={"text": "auto language"})

        from sapat.providers.openrouter import OpenRouterProvider

        provider = OpenRouterProvider()
        provider.transcribe(
            str(audio_file), model="openai/whisper-large-v3", language="auto"
        )

        _, kwargs = mock_post.call_args
        assert "language" not in kwargs["json"]
        assert kwargs["json"]["input_audio"]["format"] == "mp3"

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False)
    def test_default_model_and_config(self):
        from sapat.providers.openrouter import OpenRouterProvider

        assert OpenRouterProvider.config.default_model == "openai/whisper-large-v3"
        assert OpenRouterProvider.config.max_file_size_mb == 25.0
        assert OpenRouterProvider.config.preferred_format.value == "mp3"

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False)
    def test_resolve_model_aliases(self):
        from sapat.providers.openrouter import OpenRouterProvider

        provider = OpenRouterProvider()
        assert provider.resolve_model("whisper") == "openai/whisper-large-v3"
        assert (
            provider.resolve_model("whisper-turbo") == "openai/whisper-large-v3-turbo"
        )
        assert provider.resolve_model("gpt-4o") == "openai/gpt-4o-transcribe"
        assert provider.resolve_model("gpt-4o-mini") == "openai/gpt-4o-mini-transcribe"
        assert (
            provider.resolve_model("voxtral-mini")
            == "mistralai/voxtral-mini-transcribe"
        )
        assert provider.resolve_model("qwen-flash") == "qwen/qwen3-asr-flash-2026-02-10"
        assert provider.resolve_model("parakeet") == "nvidia/parakeet-tdt-0.6b-v3"
        assert provider.resolve_model("custom/model") == "custom/model"

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False)
    def test_unknown_extension_falls_back_to_preferred_format(self, tmp_path):
        audio_file = tmp_path / "sample.bin"
        audio_file.write_bytes(b"fake audio")

        from sapat.providers.openrouter import OpenRouterProvider

        provider = OpenRouterProvider()
        payload = provider._build_payload(
            str(audio_file),
            model="openai/whisper-large-v3",
            language="auto",
            temperature=0,
        )

        assert payload["input_audio"]["format"] == "mp3"

    @patch.dict(os.environ, {}, clear=True)
    def test_not_available_without_key(self):
        from sapat.providers.openrouter import OpenRouterProvider

        assert OpenRouterProvider.is_available() is False

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "bad-key"}, clear=False)
    @patch("sapat.providers.openrouter.requests.post")
    def test_raises_on_api_error(self, mock_post, tmp_path):
        audio_file = tmp_path / "sample.wav"
        audio_file.write_bytes(b"fake wav data")
        mock_post.return_value = FakeResponse(status_code=401, text="unauthorized")

        from sapat.providers.openrouter import OpenRouterProvider

        provider = OpenRouterProvider()
        with pytest.raises(
            RuntimeError, match="OpenRouter transcription failed \\(401\\)"
        ):
            provider.transcribe(str(audio_file), model="openai/whisper-large-v3")
