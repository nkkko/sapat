# ABOUTME: Mocked Deepgram provider tests
# ABOUTME: Verifies auth, request parameters, availability, and response parsing

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


@pytest.fixture
def audio_file(tmp_path):
    path = tmp_path / "sample.mp3"
    path.write_bytes(b"fake audio")
    return str(path)


class TestDeepgramProvider:
    @patch.dict(os.environ, {"DEEPGRAM_API_KEY": "dg-test-key"}, clear=False)
    @patch("sapat.providers.deepgram.requests.post")
    def test_transcribe_sends_token_auth_and_parses_transcript(self, mock_post, audio_file):
        mock_post.return_value = FakeResponse(
            payload={
                "metadata": {"language": "en"},
                "results": {
                    "channels": [
                        {"alternatives": [{"transcript": "Hello from Deepgram."}]}
                    ]
                },
            }
        )

        from sapat.providers.deepgram import DeepgramProvider

        provider = DeepgramProvider()
        result = provider.transcribe(audio_file, model="nova-3", language="en")

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello from Deepgram."
        assert result.language == "en"

        call = mock_post.call_args.kwargs
        assert call["headers"] == {"Authorization": "Token dg-test-key"}
        assert call["params"]["model"] == "nova-3"
        assert call["params"]["language"] == "en"
        assert call["params"]["smart_format"] == "true"
        assert call["timeout"] == 120

    @patch.dict(os.environ, {"DEEPGRAM_API_KEY": "dg-test-key"}, clear=False)
    def test_available_with_key(self):
        from sapat.providers.deepgram import DeepgramProvider

        assert DeepgramProvider.is_available() is True

    @patch.dict(os.environ, {}, clear=True)
    def test_not_available_without_key(self):
        from sapat.providers.deepgram import DeepgramProvider

        assert DeepgramProvider.is_available() is False

    @patch.dict(os.environ, {"DEEPGRAM_API_KEY": "dg-test-key"}, clear=False)
    @patch("sapat.providers.deepgram.requests.post")
    def test_non_200_response_raises(self, mock_post, audio_file):
        mock_post.return_value = FakeResponse(status_code=401, text="unauthorized")

        from sapat.providers.deepgram import DeepgramProvider

        provider = DeepgramProvider()
        with pytest.raises(RuntimeError, match="Deepgram transcription failed"):
            provider.transcribe(audio_file, model="nova-3", language="en")

    @patch.dict(os.environ, {"DEEPGRAM_API_KEY": "dg-test-key"}, clear=False)
    def test_default_model(self):
        from sapat.providers.deepgram import DeepgramProvider

        assert DeepgramProvider.config.default_model == "nova-3"
