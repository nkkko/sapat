# ABOUTME: Tests for the generic OpenAI-compatible STT provider
# ABOUTME: Verifies configurable endpoint, auth, model, and registry behavior

import os
from unittest.mock import patch

import pytest

from sapat.providers.base import TranscriptionResult


class FakeResponse:
    """Minimal fake requests.Response."""

    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self.payload = payload or {}
        self.text = text

    def json(self):
        return self.payload


@pytest.fixture
def audio_file():
    return __file__


class TestOpenAICompatibleProvider:
    @patch.dict(
        os.environ,
        {
            "OPENAI_COMPAT_STT_BASE_URL": "https://gateway.example.com/v1",
            "OPENAI_COMPAT_STT_API_KEY": "test-key",
            "OPENAI_COMPAT_STT_MODEL": "custom-whisper",
        },
        clear=False,
    )
    @patch("sapat.providers.openai_compat.requests.post")
    def test_transcribe_sends_configured_request(self, mock_post, audio_file):
        mock_post.return_value = FakeResponse(payload={"text": "hello generic"})

        from sapat.providers.openai_compatible import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider()
        model = provider.resolve_model(OpenAICompatibleProvider.config.default_model)
        result = provider.transcribe(
            audio_file,
            model=model,
            language="en",
            prompt="Product names: Sapat",
            temperature=0.1,
        )

        assert isinstance(result, TranscriptionResult)
        assert result.text == "hello generic"
        assert mock_post.call_args.args[0] == (
            "https://gateway.example.com/v1/audio/transcriptions"
        )

        _, kwargs = mock_post.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer test-key"
        assert kwargs["data"]["model"] == "custom-whisper"
        assert kwargs["data"]["language"] == "en"
        assert kwargs["data"]["prompt"] == "Product names: Sapat"
        assert kwargs["data"]["temperature"] == 0.1
        assert "file" in kwargs["files"]

    @patch.dict(
        os.environ,
        {
            "OPENAI_COMPAT_STT_BASE_URL": (
                "https://gateway.example.com/audio/transcriptions"
            ),
            "OPENAI_COMPAT_STT_API_KEY": "test-key",
            "OPENAI_COMPAT_STT_AUTH_HEADER": "api-key",
            "OPENAI_COMPAT_STT_AUTH_PREFIX": "",
        },
        clear=False,
    )
    @patch("sapat.providers.openai_compat.requests.post")
    def test_preserves_full_url_and_custom_auth_header(self, mock_post, audio_file):
        mock_post.return_value = FakeResponse(payload={"text": "custom auth"})

        from sapat.providers.openai_compatible import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider()
        provider.transcribe(audio_file, model="whisper-large-v3")

        assert mock_post.call_args.args[0] == (
            "https://gateway.example.com/audio/transcriptions"
        )
        _, kwargs = mock_post.call_args
        assert kwargs["headers"] == {"api-key": "test-key"}

    @patch.dict(
        os.environ,
        {
            "OPENAI_COMPAT_STT_BASE_URL": "https://gateway.example.com",
            "OPENAI_COMPAT_STT_API_KEY": "bad-key",
        },
        clear=False,
    )
    @patch("sapat.providers.openai_compat.requests.post")
    def test_raises_on_api_error(self, mock_post, audio_file):
        mock_post.return_value = FakeResponse(status_code=401, text="unauthorized")

        from sapat.providers.openai_compatible import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider()
        with pytest.raises(RuntimeError, match="401"):
            provider.transcribe(audio_file, model="whisper-1")

    @patch.dict(
        os.environ,
        {
            "OPENAI_COMPAT_STT_BASE_URL": "https://gateway.example.com",
            "OPENAI_COMPAT_STT_API_KEY": "test-key",
        },
        clear=False,
    )
    def test_available_with_endpoint_and_key(self):
        from sapat.providers.openai_compatible import OpenAICompatibleProvider

        assert OpenAICompatibleProvider.is_available() is True

    @patch.dict(
        os.environ,
        {"OPENAI_COMPAT_STT_BASE_URL": "https://gateway.example.com"},
        clear=True,
    )
    def test_not_available_without_key(self):
        from sapat.providers.openai_compatible import OpenAICompatibleProvider

        assert OpenAICompatibleProvider.is_available() is False

    @patch.dict(
        os.environ,
        {
            "OPENAI_COMPAT_STT_BASE_URL": "https://gateway.example.com",
            "OPENAI_COMPAT_STT_API_KEY": "test-key",
        },
        clear=False,
    )
    def test_falls_back_to_whisper_model(self):
        from sapat.providers.openai_compatible import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider()
        assert provider.resolve_model("default") == "whisper-1"
        assert provider.resolve_model("provider/model") == "provider/model"
