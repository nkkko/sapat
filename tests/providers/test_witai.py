# ABOUTME: Tests for the Wit.ai transcription provider
# ABOUTME: Verifies auth, binary upload, response parsing, and registry behavior

import os
from unittest.mock import patch

import pytest

from sapat.providers.base import TranscriptionResult


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self.payload = payload
        self.text = text

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


@pytest.fixture
def audio_file(tmp_path):
    path = tmp_path / "sample.mp3"
    path.write_bytes(b"fake audio data")
    return str(path)


class TestWitAIProvider:
    @patch.dict(os.environ, {"WITAI_ACCESS_TOKEN": "test-token"}, clear=True)
    def test_available_with_access_token(self):
        from sapat.providers.witai import WitAIProvider

        assert WitAIProvider.is_available() is True

    @patch.dict(os.environ, {}, clear=True)
    def test_not_available_without_access_token(self):
        from sapat.providers.witai import WitAIProvider

        assert WitAIProvider.is_available() is False

    @patch.dict(os.environ, {"WITAI_ACCESS_TOKEN": "test-token"}, clear=True)
    @patch("sapat.providers.witai.requests.post")
    def test_transcribe_posts_binary_audio(self, mock_post, audio_file):
        mock_post.return_value = FakeResponse(payload={"text": "hello from wit"})

        from sapat.providers.witai import WitAIProvider

        provider = WitAIProvider()
        result = provider.transcribe(audio_file, model="speech")

        assert isinstance(result, TranscriptionResult)
        assert result.text == "hello from wit"

        mock_post.assert_called_once()
        url = mock_post.call_args.args[0]
        kwargs = mock_post.call_args.kwargs

        assert url == "https://api.wit.ai/speech"
        assert kwargs["params"] == {"v": "20240304"}
        assert kwargs["headers"]["Authorization"] == "Bearer test-token"
        assert kwargs["headers"]["Content-Type"] == "audio/mpeg"
        assert kwargs["data"] == b"fake audio data"

    @patch.dict(
        os.environ,
        {
            "WITAI_ACCESS_TOKEN": "test-token",
            "WITAI_API_ENDPOINT": "https://example.test/speech",
            "WITAI_API_VERSION": "20250101",
            "WITAI_CONTENT_TYPE": "audio/wav",
        },
        clear=True,
    )
    @patch("sapat.providers.witai.requests.post")
    def test_respects_endpoint_version_and_content_type(self, mock_post, audio_file):
        mock_post.return_value = FakeResponse(payload={"text": "configured"})

        from sapat.providers.witai import WitAIProvider

        result = WitAIProvider().transcribe(audio_file, model="speech")

        assert result.text == "configured"
        assert mock_post.call_args.args[0] == "https://example.test/speech"
        assert mock_post.call_args.kwargs["params"] == {"v": "20250101"}
        assert mock_post.call_args.kwargs["headers"]["Content-Type"] == "audio/wav"

    @patch.dict(os.environ, {"WITAI_ACCESS_TOKEN": "test-token"}, clear=True)
    @patch("sapat.providers.witai.requests.post")
    def test_parses_streaming_json_response(self, mock_post, audio_file):
        mock_post.return_value = FakeResponse(
            payload=ValueError("not single json"),
            text='{"text": "partial"}\n{"text": "final transcript"}',
        )

        from sapat.providers.witai import WitAIProvider

        result = WitAIProvider().transcribe(audio_file, model="speech")

        assert result.text == "final transcript"
        assert result.raw_response == {"text": "final transcript"}

    @patch.dict(os.environ, {"WITAI_ACCESS_TOKEN": "test-token"}, clear=True)
    @patch("sapat.providers.witai.requests.post")
    def test_transcribe_api_error_raises(self, mock_post, audio_file):
        mock_post.return_value = FakeResponse(status_code=401, text="invalid token")

        from sapat.providers.witai import WitAIProvider

        with pytest.raises(RuntimeError, match="Wit.ai transcription failed"):
            WitAIProvider().transcribe(audio_file, model="speech")

    def test_decode_json_stream_handles_concatenated_objects(self):
        from sapat.providers.witai import WitAIProvider

        payload = WitAIProvider._decode_json_stream(
            '{"text": "first"}{"_text": "second"}'
        )

        assert payload == {"_text": "second"}

    def test_extract_text_rejects_empty_payload(self):
        from sapat.providers.witai import WitAIProvider

        with pytest.raises(RuntimeError, match="transcript text"):
            WitAIProvider._extract_text({"entities": {}})
