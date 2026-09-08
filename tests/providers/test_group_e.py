# ABOUTME: Tests for additional REST transcription providers
# ABOUTME: Covers Azure AI Speech short-audio REST integration

import os
from unittest.mock import patch

import pytest

from sapat.providers.base import TranscriptionResult


class FakeResponse:
    """Minimal requests.Response stand-in for mocked HTTP calls."""

    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


@pytest.fixture
def wav_audio(tmp_path):
    path = tmp_path / "sample.wav"
    path.write_bytes(b"fake wav audio")
    return str(path)


class TestAzureSpeechProvider:
    @patch.dict(
        os.environ,
        {
            "AZURE_SPEECH_KEY": "test-key",
            "AZURE_SPEECH_ENDPOINT": (
                "https://speech.example.cognitiveservices.azure.com"
            ),
        },
        clear=True,
    )
    def test_is_available_with_key_and_endpoint(self):
        from sapat.providers.azure_speech import AzureSpeechProvider

        assert AzureSpeechProvider.is_available() is True

    @patch.dict(
        os.environ,
        {
            "AZURE_SPEECH_KEY": "test-key",
            "AZURE_SPEECH_REGION": "westeurope",
        },
        clear=True,
    )
    def test_is_available_with_key_and_region_fallback(self):
        from sapat.providers.azure_speech import AzureSpeechProvider

        assert AzureSpeechProvider.is_available() is True

    @patch.dict(os.environ, {"AZURE_SPEECH_KEY": "test-key"}, clear=True)
    def test_is_not_available_without_endpoint_or_region(self):
        from sapat.providers.azure_speech import AzureSpeechProvider

        assert AzureSpeechProvider.is_available() is False

    @patch.dict(
        os.environ,
        {
            "AZURE_SPEECH_KEY": "test-key",
            "AZURE_SPEECH_REGION": "westeurope",
        },
        clear=True,
    )
    def test_resolve_model_aliases(self):
        from sapat.providers.azure_speech import AzureSpeechProvider

        provider = AzureSpeechProvider()
        assert provider.resolve_model("default") == "conversation"
        assert provider.resolve_model("conversation") == "conversation"
        assert provider.resolve_model("dictation") == "dictation"
        assert provider.resolve_model("interactive") == "interactive"

    @patch.dict(
        os.environ,
        {
            "AZURE_SPEECH_KEY": "test-key",
            "AZURE_SPEECH_ENDPOINT": (
                "https://speech.example.cognitiveservices.azure.com"
            ),
        },
        clear=True,
    )
    @patch("sapat.providers.azure_speech.requests.post")
    def test_transcribe_sends_wav_to_resource_endpoint(self, mock_post, wav_audio):
        mock_post.return_value = FakeResponse(
            200,
            {
                "RecognitionStatus": "Success",
                "DisplayText": "Hello from Azure Speech.",
                "Duration": "50000000",
            },
        )

        from sapat.providers.azure_speech import AzureSpeechProvider

        provider = AzureSpeechProvider()
        result = provider.transcribe(wav_audio, model="conversation", language="en")

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello from Azure Speech."
        assert result.language == "en-US"
        assert result.duration == 5.0

        mock_post.assert_called_once()
        url = mock_post.call_args.args[0]
        call_kwargs = mock_post.call_args.kwargs
        assert url == (
            "https://speech.example.cognitiveservices.azure.com/"
            "stt/speech/recognition/conversation/cognitiveservices/v1"
        )
        assert call_kwargs["params"] == {"language": "en-US", "format": "simple"}
        assert call_kwargs["headers"]["Ocp-Apim-Subscription-Key"] == "test-key"
        assert call_kwargs["headers"]["Content-Type"] == (
            "audio/wav; codecs=audio/pcm; samplerate=16000"
        )
        assert call_kwargs["data"] == b"fake wav audio"

    @patch.dict(
        os.environ,
        {
            "AZURE_SPEECH_KEY": "test-key",
            "AZURE_SPEECH_REGION": "westeurope",
            "AZURE_SPEECH_RESPONSE_FORMAT": "detailed",
            "AZURE_SPEECH_PROFANITY": "raw",
        },
        clear=True,
    )
    @patch("sapat.providers.azure_speech.requests.post")
    def test_transcribe_uses_detailed_response_fields(self, mock_post, wav_audio):
        mock_post.return_value = FakeResponse(
            200,
            {
                "RecognitionStatus": "Success",
                "NBest": [{"Display": "Detailed transcript."}],
            },
        )

        from sapat.providers.azure_speech import AzureSpeechProvider

        provider = AzureSpeechProvider()
        result = provider.transcribe(wav_audio, model="default", language="bg")

        assert result.text == "Detailed transcript."
        assert mock_post.call_args.kwargs["params"] == {
            "language": "bg-BG",
            "format": "detailed",
            "profanity": "raw",
        }

    @patch.dict(
        os.environ,
        {
            "AZURE_SPEECH_KEY": "test-key",
            "AZURE_SPEECH_REGION": "westeurope",
            "AZURE_SPEECH_ENDPOINT": "https://speech.example.cognitiveservices.azure.com",
        },
        clear=True,
    )
    def test_builds_resource_endpoint_with_stt_prefix(self):
        from sapat.providers.azure_speech import AzureSpeechProvider

        provider = AzureSpeechProvider()
        assert provider._build_endpoint("conversation") == (
            "https://speech.example.cognitiveservices.azure.com/"
            "stt/speech/recognition/conversation/cognitiveservices/v1"
        )

    @patch.dict(
        os.environ,
        {
            "AZURE_SPEECH_KEY": "test-key",
            "AZURE_SPEECH_REGION": "westeurope",
        },
        clear=True,
    )
    def test_builds_regional_fallback_endpoint(self):
        from sapat.providers.azure_speech import AzureSpeechProvider

        provider = AzureSpeechProvider()
        assert provider._build_endpoint("conversation") == (
            "https://westeurope.stt.speech.microsoft.com/"
            "speech/recognition/conversation/cognitiveservices/v1"
        )

    @patch.dict(
        os.environ,
        {
            "AZURE_SPEECH_KEY": "test-key",
            "AZURE_SPEECH_REGION": "westeurope",
        },
        clear=True,
    )
    @patch("sapat.providers.azure_speech.requests.post")
    def test_http_error_raises_message_from_payload(self, mock_post, wav_audio):
        mock_post.return_value = FakeResponse(
            401,
            {"error": {"message": "invalid key"}},
            text="Unauthorized",
        )

        from sapat.providers.azure_speech import AzureSpeechProvider

        provider = AzureSpeechProvider()
        with pytest.raises(RuntimeError, match="invalid key"):
            provider.transcribe(wav_audio, model="conversation", language="en-US")

    @patch.dict(
        os.environ,
        {
            "AZURE_SPEECH_KEY": "test-key",
            "AZURE_SPEECH_REGION": "westeurope",
        },
        clear=True,
    )
    @patch("sapat.providers.azure_speech.requests.post")
    def test_no_match_returns_empty_transcript(self, mock_post, wav_audio):
        mock_post.return_value = FakeResponse(200, {"RecognitionStatus": "NoMatch"})

        from sapat.providers.azure_speech import AzureSpeechProvider

        provider = AzureSpeechProvider()
        result = provider.transcribe(wav_audio, model="conversation", language="en-US")

        assert result.text == ""
        assert result.raw_response == {"RecognitionStatus": "NoMatch"}

    @patch.dict(
        os.environ,
        {
            "AZURE_SPEECH_KEY": "test-key",
            "AZURE_SPEECH_REGION": "westeurope",
            "AZURE_SPEECH_RESPONSE_FORMAT": "xml",
        },
        clear=True,
    )
    def test_invalid_response_format_raises(self):
        from sapat.providers.azure_speech import AzureSpeechProvider

        provider = AzureSpeechProvider()
        with pytest.raises(ValueError, match="AZURE_SPEECH_RESPONSE_FORMAT"):
            provider._build_params("en")
