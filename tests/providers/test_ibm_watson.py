# ABOUTME: Tests for IBM Watson Speech to Text provider
# ABOUTME: Verifies auth, payload shape, response parsing, and availability gates

import os
from unittest.mock import patch

import pytest

from sapat.providers.base import TranscriptionResult


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


@pytest.fixture
def audio_file(tmp_path):
    path = tmp_path / "sample.wav"
    path.write_bytes(b"fake wav bytes")
    return str(path)


class TestIBMWatsonProvider:
    @patch.dict(
        os.environ,
        {
            "IBM_WATSON_STT_API_KEY": "test-key",
            "IBM_WATSON_STT_URL": "https://ibm.example.com/instance",
        },
        clear=True,
    )
    def test_available_with_required_config(self):
        from sapat.providers.ibm_watson import IBMWatsonProvider

        assert IBMWatsonProvider.is_available() is True

    @patch.dict(os.environ, {"IBM_WATSON_STT_API_KEY": "test-key"}, clear=True)
    def test_not_available_without_service_url(self):
        from sapat.providers.ibm_watson import IBMWatsonProvider

        assert IBMWatsonProvider.is_available() is False

    @patch.dict(os.environ, {}, clear=True)
    def test_not_available_without_key(self):
        from sapat.providers.ibm_watson import IBMWatsonProvider

        assert IBMWatsonProvider.is_available() is False

    @patch.dict(
        os.environ,
        {
            "IBM_WATSON_STT_API_KEY": "test-key",
            "IBM_WATSON_STT_URL": "https://ibm.example.com/instance/",
        },
        clear=True,
    )
    @patch("sapat.providers.ibm_watson.requests.post")
    def test_transcribe_sends_binary_recognize_request(self, mock_post, audio_file):
        mock_post.return_value = FakeResponse(
            payload={
                "results": [
                    {"alternatives": [{"transcript": "hello "}]},
                    {"alternatives": [{"transcript": "world"}]},
                ]
            }
        )

        from sapat.providers.ibm_watson import IBMWatsonProvider

        provider = IBMWatsonProvider()
        result = provider.transcribe(audio_file, model="en-US_BroadbandModel")

        assert isinstance(result, TranscriptionResult)
        assert result.text == "hello\nworld"

        assert (
            mock_post.call_args.args[0]
            == "https://ibm.example.com/instance/v1/recognize"
        )
        kwargs = mock_post.call_args.kwargs
        assert kwargs["auth"] == ("apikey", "test-key")
        assert kwargs["headers"] == {"Content-Type": "audio/wav"}
        assert kwargs["params"] == {"model": "en-US_BroadbandModel"}
        assert kwargs["timeout"] == 120.0

    @patch.dict(
        os.environ,
        {
            "IBM_WATSON_STT_API_KEY": "test-key",
            "IBM_WATSON_STT_URL": "https://ibm.example.com/instance",
        },
        clear=True,
    )
    @patch("sapat.providers.ibm_watson.requests.post")
    def test_raises_on_api_error(self, mock_post, audio_file):
        mock_post.return_value = FakeResponse(status_code=401, text="unauthorized")

        from sapat.providers.ibm_watson import IBMWatsonProvider

        provider = IBMWatsonProvider()
        with pytest.raises(RuntimeError, match="401"):
            provider.transcribe(audio_file, model="en-US_BroadbandModel")

    def test_extract_transcript_requires_text(self):
        from sapat.providers.ibm_watson import IBMWatsonProvider

        with pytest.raises(RuntimeError, match="no transcript"):
            IBMWatsonProvider._extract_transcript({"results": []})

    def test_content_type_from_suffix(self):
        from sapat.providers.ibm_watson import IBMWatsonProvider

        assert IBMWatsonProvider._content_type("clip.mp3") == "audio/mp3"
        assert IBMWatsonProvider._content_type("clip.wav") == "audio/wav"
        assert (
            IBMWatsonProvider._content_type("clip.unknown")
            == "application/octet-stream"
        )

    @patch.dict(
        os.environ,
        {
            "IBM_WATSON_STT_API_KEY": "test-key",
            "IBM_WATSON_STT_URL": "https://ibm.example.com/instance",
        },
        clear=True,
    )
    def test_model_aliases(self):
        from sapat.providers.ibm_watson import IBMWatsonProvider

        provider = IBMWatsonProvider()
        assert provider.resolve_model("en") == "en-US_BroadbandModel"
        assert provider.resolve_model("en-us") == "en-US_BroadbandModel"
        assert provider.resolve_model("custom-model") == "custom-model"
