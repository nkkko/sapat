# ABOUTME: Mock-based tests for transcription providers in group A
# ABOUTME: Tests verify each provider sends correct auth, URL, and payload

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from sapat.providers.base import TranscriptionResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def audio_file(tmp_path):
    """Create a temporary MP3 audio file for testing."""
    path = tmp_path / "test.mp3"
    path.write_bytes(b"fake audio data")
    return str(path)


class FakeResponse:
    """Minimal fake requests.Response."""

    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self.payload = payload or {}
        self.text = text

    def json(self):
        return self.payload


# ===========================================================================
# OpenAI
# ===========================================================================


class TestOpenAIProvider:
    @patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_TRANSCRIPTION_ENDPOINT": "https://example.test/v1/audio/transcriptions",
        },
        clear=False,
    )
    @patch("sapat.providers.openai.requests.post")
    def test_transcribe_sends_correct_request(self, mock_post, audio_file):
        mock_post.return_value = FakeResponse(payload={"text": "hello openai"})

        from sapat.providers.openai import OpenAIProvider

        provider = OpenAIProvider()
        result = provider.transcribe(
            audio_file,
            model="gpt4o",
            language="en",
            prompt="Product names include SAPAT.",
            temperature=0,
        )

        assert isinstance(result, TranscriptionResult)
        assert result.text == "hello openai"

        _, kwargs = mock_post.call_args
        assert (
            mock_post.call_args.args[0]
            == "https://example.test/v1/audio/transcriptions"
        )
        assert kwargs["headers"]["Authorization"] == "Bearer test-key"
        assert kwargs["data"]["model"] == "gpt-4o-transcribe"
        assert kwargs["data"]["language"] == "en"
        assert kwargs["data"]["prompt"] == "Product names include SAPAT."
        assert kwargs["data"]["response_format"] == "json"
        assert "file" in kwargs["files"]

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    @patch("sapat.providers.openai.requests.post")
    def test_diarization_defaults_to_speaker_segments(self, mock_post, audio_file):
        segments = [{"speaker": "speaker_0", "text": "hello", "start": 0, "end": 1}]
        mock_post.return_value = FakeResponse(
            payload={"text": "hello", "segments": segments, "duration": 1.0}
        )

        from sapat.providers.openai import OpenAIProvider

        provider = OpenAIProvider()
        result = provider.transcribe(audio_file, model="diarize")

        _, kwargs = mock_post.call_args
        assert kwargs["data"]["model"] == "gpt-4o-transcribe-diarize"
        assert kwargs["data"]["response_format"] == "diarized_json"
        assert kwargs["data"]["chunking_strategy"] == "auto"
        assert result.segments == segments
        assert result.raw_response["duration"] == 1.0

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    def test_resolve_model_aliases(self):
        from sapat.providers.openai import OpenAIProvider

        provider = OpenAIProvider()
        assert provider.resolve_model("whisper") == "whisper-1"
        assert provider.resolve_model("mini") == "gpt-4o-mini-transcribe"
        assert provider.resolve_model("diarization") == "gpt-4o-transcribe-diarize"

    @patch.dict(os.environ, {}, clear=True)
    def test_not_available_without_key(self):
        from sapat.providers.openai import OpenAIProvider

        assert OpenAIProvider.is_available() is False

    @patch.dict(os.environ, {"OPENAI_API_KEY": "bad-key"}, clear=False)
    @patch("sapat.providers.openai.requests.post")
    def test_raises_on_api_error(self, mock_post, audio_file):
        mock_post.return_value = FakeResponse(status_code=401, text="unauthorized")

        from sapat.providers.openai import OpenAIProvider

        provider = OpenAIProvider()
        with pytest.raises(RuntimeError, match="401"):
            provider.transcribe(audio_file, model="gpt-4o-transcribe")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    def test_diarization_rejects_prompt(self, audio_file):
        from sapat.providers.openai import OpenAIProvider

        provider = OpenAIProvider()
        with pytest.raises(ValueError, match="do not support prompts"):
            provider.transcribe(audio_file, model="diarize", prompt="Prefer names")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    def test_diarization_rejects_unsupported_options(self, audio_file):
        from sapat.providers.openai import OpenAIProvider

        provider = OpenAIProvider()
        with pytest.raises(ValueError, match="timestamp_granularities"):
            provider.transcribe(
                audio_file,
                model="diarize",
                timestamp_granularities=["word"],
            )


# ===========================================================================
# 1. DeepInfra
# ===========================================================================


class TestDeepInfraProvider:
    @patch.dict(os.environ, {"DEEPINFRA_API_KEY": "test-key"}, clear=False)
    @patch("sapat.providers.openai_compat.requests.post")
    def test_transcribe_sends_correct_request(self, mock_post, audio_file):
        mock_post.return_value = FakeResponse(payload={"text": "hello deepinfra"})

        from sapat.providers.deepinfra import DeepInfraProvider

        provider = DeepInfraProvider()
        result = provider.transcribe(audio_file, model="openai/whisper-large-v3")

        assert isinstance(result, TranscriptionResult)
        assert result.text == "hello deepinfra"

        _, kwargs = mock_post.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer test-key"
        assert (
            mock_post.call_args.args[0]
            == "https://api.deepinfra.com/v1/openai/audio/transcriptions"
        )
        assert kwargs["data"]["model"] == "openai/whisper-large-v3"
        assert "file" in kwargs["files"]

    @patch.dict(os.environ, {"DEEPINFRA_API_KEY": "test-key"}, clear=False)
    def test_default_model(self):
        from sapat.providers.deepinfra import DeepInfraProvider

        assert DeepInfraProvider.config.default_model == "openai/whisper-large-v3"

    @patch.dict(os.environ, {}, clear=True)
    def test_not_available_without_key(self):
        from sapat.providers.deepinfra import DeepInfraProvider

        assert DeepInfraProvider.is_available() is False

    @patch.dict(os.environ, {"DEEPINFRA_API_KEY": "bad-key"}, clear=False)
    @patch("sapat.providers.openai_compat.requests.post")
    def test_raises_on_api_error(self, mock_post, audio_file):
        mock_post.return_value = FakeResponse(status_code=401, text="unauthorized")

        from sapat.providers.deepinfra import DeepInfraProvider

        provider = DeepInfraProvider()
        with pytest.raises(RuntimeError, match="401"):
            provider.transcribe(audio_file, model="openai/whisper-large-v3")


# ===========================================================================
# 2. Venice
# ===========================================================================


class TestVeniceProvider:
    @patch.dict(os.environ, {"VENICE_API_KEY": "test-key"}, clear=False)
    @patch("sapat.providers.openai_compat.requests.post")
    def test_transcribe_sends_correct_request(self, mock_post, audio_file):
        mock_post.return_value = FakeResponse(payload={"text": "hello venice"})

        from sapat.providers.venice import VeniceProvider

        provider = VeniceProvider()
        result = provider.transcribe(audio_file, model="whisper-large-v3")

        assert isinstance(result, TranscriptionResult)
        assert result.text == "hello venice"

        _, kwargs = mock_post.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer test-key"
        assert (
            mock_post.call_args.args[0]
            == "https://api.venice.ai/api/v1/audio/transcriptions"
        )
        assert kwargs["data"]["model"] == "whisper-large-v3"
        assert "file" in kwargs["files"]

    @patch.dict(os.environ, {"VENICE_API_KEY": "test-key"}, clear=False)
    def test_default_model(self):
        from sapat.providers.venice import VeniceProvider

        assert VeniceProvider.config.default_model == "whisper-large-v3"

    @patch.dict(os.environ, {}, clear=True)
    def test_not_available_without_key(self):
        from sapat.providers.venice import VeniceProvider

        assert VeniceProvider.is_available() is False


# ===========================================================================
# 3. Together AI
# ===========================================================================


class TestTogetherProvider:
    @patch.dict(os.environ, {"TOGETHER_API_KEY": "test-key"}, clear=False)
    @patch("sapat.providers.openai_compat.requests.post")
    def test_transcribe_sends_correct_request(self, mock_post, audio_file):
        mock_post.return_value = FakeResponse(payload={"text": "hello together"})

        from sapat.providers.together import TogetherProvider

        provider = TogetherProvider()
        result = provider.transcribe(audio_file, model="openai/whisper-large-v3")

        assert isinstance(result, TranscriptionResult)
        assert result.text == "hello together"

        _, kwargs = mock_post.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer test-key"
        assert (
            mock_post.call_args.args[0]
            == "https://api.together.xyz/v1/audio/transcriptions"
        )
        assert kwargs["data"]["model"] == "openai/whisper-large-v3"
        assert "file" in kwargs["files"]

    @patch.dict(os.environ, {"TOGETHER_API_KEY": "test-key"}, clear=False)
    def test_default_model(self):
        from sapat.providers.together import TogetherProvider

        assert TogetherProvider.config.default_model == "openai/whisper-large-v3"

    @patch.dict(os.environ, {}, clear=True)
    def test_not_available_without_key(self):
        from sapat.providers.together import TogetherProvider

        assert TogetherProvider.is_available() is False


# ===========================================================================
# 4. xAI
# ===========================================================================


class TestXAIProvider:
    @patch.dict(os.environ, {"XAI_API_KEY": "test-key"}, clear=False)
    @patch("sapat.providers.openai_compat.requests.post")
    def test_transcribe_sends_correct_request(self, mock_post, audio_file):
        mock_post.return_value = FakeResponse(payload={"text": "hello xai"})

        from sapat.providers.xai import XAIProvider

        provider = XAIProvider()
        result = provider.transcribe(audio_file, model="whisper-large-v3")

        assert isinstance(result, TranscriptionResult)
        assert result.text == "hello xai"

        _, kwargs = mock_post.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer test-key"
        assert mock_post.call_args.args[0] == "https://api.x.ai/v1/audio/transcriptions"
        assert kwargs["data"]["model"] == "whisper-large-v3"
        assert "file" in kwargs["files"]

    @patch.dict(os.environ, {"XAI_API_KEY": "test-key"}, clear=False)
    def test_default_model(self):
        from sapat.providers.xai import XAIProvider

        assert XAIProvider.config.default_model == "whisper-large-v3"

    @patch.dict(os.environ, {}, clear=True)
    def test_not_available_without_key(self):
        from sapat.providers.xai import XAIProvider

        assert XAIProvider.is_available() is False


# ===========================================================================
# 5. Mistral
# ===========================================================================


class TestMistralProvider:
    @patch.dict(os.environ, {"MISTRAL_API_KEY": "test-key"}, clear=False)
    @patch("sapat.providers.mistral.requests.post")
    def test_transcribe_sends_correct_request(self, mock_post, audio_file):
        mock_post.return_value = FakeResponse(payload={"text": "hello mistral"})

        from sapat.providers.mistral import MistralProvider

        provider = MistralProvider()
        result = provider.transcribe(audio_file, model="voxtral-mini-latest")

        assert isinstance(result, TranscriptionResult)
        assert result.text == "hello mistral"

        _, kwargs = mock_post.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer test-key"
        assert (
            mock_post.call_args.args[0]
            == "https://api.mistral.ai/v1/audio/transcriptions"
        )
        assert kwargs["data"]["model"] == "voxtral-mini-latest"
        # Mistral uses context_bias instead of prompt
        assert "prompt" not in kwargs["data"]
        assert "file" in kwargs["files"]

    @patch.dict(os.environ, {"MISTRAL_API_KEY": "test-key"}, clear=False)
    @patch("sapat.providers.mistral.requests.post")
    def test_transcribe_sends_context_bias_for_prompt(self, mock_post, audio_file):
        mock_post.return_value = FakeResponse(payload={"text": "hello"})

        from sapat.providers.mistral import MistralProvider

        provider = MistralProvider()
        provider.transcribe(
            audio_file, model="voxtral-mini-latest", prompt="Product: Sapat"
        )

        _, kwargs = mock_post.call_args
        assert kwargs["data"]["context_bias"] == "Product: Sapat"

    @patch.dict(os.environ, {"MISTRAL_API_KEY": "test-key"}, clear=False)
    @patch("sapat.providers.mistral.requests.post")
    def test_correct_transcript_uses_chat_endpoint(self, mock_post):
        chat_response = FakeResponse(
            payload={"choices": [{"message": {"content": "corrected text"}}]}
        )
        mock_post.return_value = chat_response

        from sapat.providers.mistral import MistralProvider

        provider = MistralProvider()
        result = provider.correct_transcript("raw text", temperature=0.1)

        assert result == "corrected text"
        _, kwargs = mock_post.call_args
        assert (
            mock_post.call_args.args[0] == "https://api.mistral.ai/v1/chat/completions"
        )
        assert kwargs["json"]["messages"][1]["content"] == "raw text"

    @patch.dict(os.environ, {"MISTRAL_API_KEY": "test-key"}, clear=False)
    def test_supports_correction(self):
        from sapat.providers.mistral import MistralProvider

        assert MistralProvider.config.supports_correction is True

    @patch.dict(os.environ, {"MISTRAL_API_KEY": "test-key"}, clear=False)
    def test_default_model(self):
        from sapat.providers.mistral import MistralProvider

        assert MistralProvider.config.default_model == "voxtral-mini-latest"

    @patch.dict(os.environ, {}, clear=True)
    def test_not_available_without_key(self):
        from sapat.providers.mistral import MistralProvider

        assert MistralProvider.is_available() is False


# ===========================================================================
# 6. Lemonfox
# ===========================================================================


class TestLemonfoxProvider:
    @patch.dict(os.environ, {"LEMONFOX_API_KEY": "test-key"}, clear=False)
    @patch("sapat.providers.openai_compat.requests.post")
    def test_transcribe_sends_correct_request(self, mock_post, audio_file):
        mock_post.return_value = FakeResponse(payload={"text": "hello lemonfox"})

        from sapat.providers.lemonfox import LemonfoxProvider

        provider = LemonfoxProvider()
        result = provider.transcribe(audio_file, model="whisper-1")

        assert isinstance(result, TranscriptionResult)
        assert result.text == "hello lemonfox"

        _, kwargs = mock_post.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer test-key"
        assert (
            mock_post.call_args.args[0]
            == "https://api.lemonfox.ai/v1/audio/transcriptions"
        )
        assert kwargs["data"]["model"] == "whisper-1"
        assert "file" in kwargs["files"]

    @patch.dict(os.environ, {"LEMONFOX_API_KEY": "test-key"}, clear=False)
    def test_default_model(self):
        from sapat.providers.lemonfox import LemonfoxProvider

        assert LemonfoxProvider.config.default_model == "whisper-1"

    @patch.dict(os.environ, {}, clear=True)
    def test_not_available_without_key(self):
        from sapat.providers.lemonfox import LemonfoxProvider

        assert LemonfoxProvider.is_available() is False


# ===========================================================================
# 7. LocalAI
# ===========================================================================


class TestLocalAIProvider:
    @patch.dict(
        os.environ,
        {"LOCALAI_BASE_URL": "http://localhost:8080"},
        clear=False,
    )
    @patch("sapat.providers.localai.requests.post")
    def test_transcribe_sends_correct_request(self, mock_post, audio_file):
        mock_post.return_value = FakeResponse(payload={"text": "hello localai"})

        from sapat.providers.localai import LocalAIProvider

        provider = LocalAIProvider()
        result = provider.transcribe(audio_file, model="whisper-1")

        assert isinstance(result, TranscriptionResult)
        assert result.text == "hello localai"

        _, kwargs = mock_post.call_args
        # No auth header when LOCALAI_API_KEY is not set
        assert "Authorization" not in kwargs["headers"]
        assert (
            mock_post.call_args.args[0]
            == "http://localhost:8080/v1/audio/transcriptions"
        )
        assert kwargs["data"]["model"] == "whisper-1"
        assert "file" in kwargs["files"]

    @patch.dict(
        os.environ,
        {
            "LOCALAI_BASE_URL": "http://localhost:8080",
            "LOCALAI_API_KEY": "my-token",
        },
        clear=False,
    )
    @patch("sapat.providers.localai.requests.post")
    def test_includes_auth_header_when_key_set(self, mock_post, audio_file):
        mock_post.return_value = FakeResponse(payload={"text": "hello"})

        from sapat.providers.localai import LocalAIProvider

        provider = LocalAIProvider()
        provider.transcribe(audio_file, model="whisper-1")

        _, kwargs = mock_post.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer my-token"

    @patch.dict(
        os.environ,
        {"LOCALAI_BASE_URL": "http://localhost:8080"},
        clear=False,
    )
    def test_default_model(self):
        from sapat.providers.localai import LocalAIProvider

        assert LocalAIProvider.config.default_model == "whisper-1"

    @patch.dict(os.environ, {}, clear=True)
    def test_not_available_without_base_url(self):
        from sapat.providers.localai import LocalAIProvider

        assert LocalAIProvider.is_available() is False


# ===========================================================================
# 8. ElevenLabs
# ===========================================================================


class TestElevenLabsProvider:
    @patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test-key"}, clear=False)
    @patch("sapat.providers.elevenlabs.requests.post")
    def test_transcribe_sends_xi_api_key_header(self, mock_post, audio_file):
        mock_post.return_value = FakeResponse(payload={"text": "hello elevenlabs"})

        from sapat.providers.elevenlabs import ElevenLabsProvider

        provider = ElevenLabsProvider()
        result = provider.transcribe(audio_file, model="scribe_v2")

        assert isinstance(result, TranscriptionResult)
        assert result.text == "hello elevenlabs"

        _, kwargs = mock_post.call_args
        # Must use xi-api-key, NOT Authorization Bearer
        assert "xi-api-key" in kwargs["headers"]
        assert kwargs["headers"]["xi-api-key"] == "test-key"
        assert "Authorization" not in kwargs["headers"]
        assert (
            mock_post.call_args.args[0] == "https://api.elevenlabs.io/v1/speech-to-text"
        )
        # ElevenLabs uses model_id, not model
        assert kwargs["data"]["model_id"] == "scribe_v2"
        assert "file" in kwargs["files"]

    @patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test-key"}, clear=False)
    @patch("sapat.providers.elevenlabs.requests.post")
    def test_sends_language_code_not_language(self, mock_post, audio_file):
        mock_post.return_value = FakeResponse(payload={"text": "hello"})

        from sapat.providers.elevenlabs import ElevenLabsProvider

        provider = ElevenLabsProvider()
        provider.transcribe(audio_file, model="scribe_v2", language="en")

        _, kwargs = mock_post.call_args
        assert kwargs["data"]["language_code"] == "en"
        assert "language" not in kwargs["data"]

    @patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test-key"}, clear=False)
    def test_default_model(self):
        from sapat.providers.elevenlabs import ElevenLabsProvider

        assert ElevenLabsProvider.config.default_model == "scribe_v2"

    @patch.dict(os.environ, {}, clear=True)
    def test_not_available_without_key(self):
        from sapat.providers.elevenlabs import ElevenLabsProvider

        assert ElevenLabsProvider.is_available() is False

    @patch.dict(os.environ, {"ELEVENLABS_API_KEY": "bad-key"}, clear=False)
    @patch("sapat.providers.elevenlabs.requests.post")
    def test_raises_on_api_error(self, mock_post, audio_file):
        mock_post.return_value = FakeResponse(status_code=401, text="invalid api key")

        from sapat.providers.elevenlabs import ElevenLabsProvider

        provider = ElevenLabsProvider()
        with pytest.raises(RuntimeError, match="401"):
            provider.transcribe(audio_file, model="scribe_v2")
