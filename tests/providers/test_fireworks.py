# ABOUTME: Tests for the Fireworks AI transcription provider
# ABOUTME: Verifies request shape, model routing, response parsing, and availability

import os
from unittest.mock import patch

import pytest

from sapat.providers.base import TranscriptionResult


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        return self._payload


@pytest.fixture
def audio_file(tmp_path):
    path = tmp_path / "sample.wav"
    path.write_bytes(b"fake wav bytes")
    return str(path)


class TestFireworksProvider:
    @patch.dict(os.environ, {"FIREWORKS_API_KEY": "fw-test-key"}, clear=True)
    def test_available_with_api_key(self):
        from sapat.providers.fireworks import FireworksProvider

        assert FireworksProvider.is_available() is True

    @patch.dict(os.environ, {}, clear=True)
    def test_not_available_without_api_key(self):
        from sapat.providers.fireworks import FireworksProvider

        assert FireworksProvider.is_available() is False

    @patch.dict(os.environ, {"FIREWORKS_API_KEY": "fw-test-key"}, clear=True)
    @patch("sapat.providers.fireworks.requests.post")
    def test_transcribe_sends_prod_request(self, mock_post, audio_file):
        mock_post.return_value = FakeResponse(
            payload={"text": "hello fireworks", "language": "en"}
        )

        from sapat.providers.fireworks import FireworksProvider

        provider = FireworksProvider()
        result = provider.transcribe(
            audio_file,
            model="whisper-v3",
            language="en",
            prompt="product names: Daytona, Sapat",
            temperature=0.2,
        )

        assert isinstance(result, TranscriptionResult)
        assert result.text == "hello fireworks"
        assert result.language == "en"

        assert (
            mock_post.call_args.args[0]
            == "https://audio-prod.api.fireworks.ai/v1/audio/transcriptions"
        )
        kwargs = mock_post.call_args.kwargs
        assert kwargs["headers"] == {"Authorization": "fw-test-key"}
        assert kwargs["data"]["model"] == "whisper-v3"
        assert kwargs["data"]["response_format"] == "json"
        assert kwargs["data"]["language"] == "en"
        assert kwargs["data"]["prompt"] == "product names: Daytona, Sapat"
        assert kwargs["data"]["temperature"] == 0.2
        assert "file" in kwargs["files"]

    @patch.dict(os.environ, {"FIREWORKS_API_KEY": "fw-test-key"}, clear=True)
    @patch("sapat.providers.fireworks.requests.post")
    def test_transcribe_routes_turbo_model_to_turbo_endpoint(
        self, mock_post, audio_file
    ):
        mock_post.return_value = FakeResponse(payload={"text": "fast transcript"})

        from sapat.providers.fireworks import FireworksProvider

        provider = FireworksProvider()
        result = provider.transcribe(audio_file, model="whisper-v3-turbo")

        assert result.text == "fast transcript"
        assert (
            mock_post.call_args.args[0]
            == "https://audio-turbo.api.fireworks.ai/v1/audio/transcriptions"
        )

    @patch.dict(os.environ, {"FIREWORKS_API_KEY": "fw-test-key"}, clear=True)
    @patch("sapat.providers.fireworks.requests.post")
    def test_raises_on_api_error(self, mock_post, audio_file):
        mock_post.return_value = FakeResponse(status_code=401, text="unauthorized")

        from sapat.providers.fireworks import FireworksProvider

        provider = FireworksProvider()
        with pytest.raises(RuntimeError, match="401"):
            provider.transcribe(audio_file, model="whisper-v3")

    def test_raises_when_response_has_no_text(self):
        from sapat.providers.fireworks import FireworksProvider

        with pytest.raises(RuntimeError, match="no transcript"):
            FireworksProvider._extract_text({})

    def test_model_aliases(self):
        from sapat.providers.fireworks import FireworksProvider

        provider = FireworksProvider()
        assert provider.resolve_model("default") == "whisper-v3"
        assert provider.resolve_model("whisper") == "whisper-v3"
        assert provider.resolve_model("turbo") == "whisper-v3-turbo"
        assert provider.resolve_model("custom-model") == "custom-model"
