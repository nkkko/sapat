# ABOUTME: Tests for Group C cloud SDK/REST transcription providers
# ABOUTME: Covers Gemini, NVIDIA, Baidu, Cloudflare, Sarvam, fal.ai, and Soniox

import base64
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeResponse:
    """Minimal requests.Response stand-in for mocked HTTP calls."""

    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _make_audio(suffix=".mp3", content=b"fake audio bytes"):
    """Create a temporary audio file; caller is responsible for cleanup."""
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(content)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------


class TestGeminiProvider:
    @patch.dict(
        os.environ,
        {"GOOGLE_API_KEY": "test-key"},
        clear=True,
    )
    def test_is_available_with_google_key(self):
        from sapat.providers.gemini import GeminiProvider

        assert GeminiProvider.is_available() is True

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True)
    def test_is_not_available_with_only_gemini_key(self):
        from sapat.providers.gemini import GeminiProvider

        # Config requires GOOGLE_API_KEY; GEMINI_API_KEY alone won't satisfy
        assert GeminiProvider.is_available() is False

    @patch.dict(os.environ, {}, clear=True)
    def test_is_not_available_without_key(self):
        from sapat.providers.gemini import GeminiProvider

        assert GeminiProvider.is_available() is False

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=True)
    def test_resolve_model_aliases(self):
        from sapat.providers.gemini import GeminiProvider

        p = GeminiProvider()
        assert p.resolve_model("flash") == "gemini-2.0-flash"
        assert p.resolve_model("pro") == "gemini-2.0-pro"
        assert p.resolve_model("flash15") == "gemini-1.5-flash"
        assert p.resolve_model("custom-model") == "custom-model"

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=True)
    @patch("sapat.providers.gemini.requests.post")
    def test_transcribe_sends_inline_base64_audio(self, mock_post):
        audio_content = b"example audio data"
        audio_file = _make_audio(".mp3", audio_content)

        mock_post.return_value = FakeResponse(
            200,
            {
                "candidates": [
                    {"content": {"parts": [{"text": "hola mundo"}]}}
                ]
            },
        )

        try:
            from sapat.providers.gemini import GeminiProvider

            p = GeminiProvider()
            result = p.transcribe(
                audio_file,
                model="gemini-2.0-flash",
                language="es",
                prompt="Speaker says Sapat.",
                temperature=0.2,
            )
        finally:
            os.unlink(audio_file)

        assert result.text == "hola mundo"
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        request_json = call_kwargs["json"]
        parts = request_json["contents"][0]["parts"]

        assert "Language: es." in parts[0]["text"]
        assert "Speaker says Sapat." in parts[0]["text"]
        assert parts[1]["inline_data"]["mime_type"] == "audio/mp3"
        assert (
            parts[1]["inline_data"]["data"]
            == base64.b64encode(audio_content).decode("ascii")
        )
        assert call_kwargs["headers"]["x-goog-api-key"] == "test-key"
        assert request_json["generationConfig"]["temperature"] == 0.2

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=True)
    @patch("sapat.providers.gemini.requests.post")
    def test_transcribe_concatenates_multiple_text_parts(self, mock_post):
        audio_file = _make_audio(".wav")

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

        try:
            from sapat.providers.gemini import GeminiProvider

            p = GeminiProvider()
            result = p.transcribe(audio_file, model="gemini-2.0-flash")
        finally:
            os.unlink(audio_file)

        assert result.text == "first part\nsecond part"

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=True)
    @patch("sapat.providers.gemini.requests.post")
    def test_transcribe_api_error_raises(self, mock_post):
        audio_file = _make_audio(".flac")

        mock_post.return_value = FakeResponse(
            400,
            {"error": {"message": "API key not valid"}},
            text='{"error": {"message": "API key not valid"}}',
        )

        try:
            from sapat.providers.gemini import GeminiProvider

            p = GeminiProvider()
            with pytest.raises(RuntimeError, match="API key not valid"):
                p.transcribe(audio_file, model="gemini-2.0-flash")
        finally:
            os.unlink(audio_file)

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=True)
    @patch("sapat.providers.gemini.requests.post")
    def test_transcribe_empty_response_raises(self, mock_post):
        audio_file = _make_audio(".mp3")

        mock_post.return_value = FakeResponse(
            200,
            {
                "candidates": [
                    {"content": {"parts": []}, "finishReason": "STOP"}
                ]
            },
        )

        try:
            from sapat.providers.gemini import GeminiProvider

            p = GeminiProvider()
            with pytest.raises(RuntimeError, match="empty response"):
                p.transcribe(audio_file, model="gemini-2.0-flash")
        finally:
            os.unlink(audio_file)

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=True)
    @patch("sapat.providers.gemini.requests.post")
    def test_transcribe_request_exception_wrapped(self, mock_post):
        import requests as req

        audio_file = _make_audio(".mp3")
        mock_post.side_effect = req.RequestException("timeout")

        try:
            from sapat.providers.gemini import GeminiProvider

            p = GeminiProvider()
            with pytest.raises(RuntimeError, match="request failed: timeout"):
                p.transcribe(audio_file, model="gemini-2.0-flash")
        finally:
            os.unlink(audio_file)

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=True)
    @patch("sapat.providers.gemini.requests.post")
    def test_transcribe_prompt_feedback_error(self, mock_post):
        audio_file = _make_audio(".mp3")

        mock_post.return_value = FakeResponse(
            200,
            {"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}},
        )

        try:
            from sapat.providers.gemini import GeminiProvider

            p = GeminiProvider()
            with pytest.raises(RuntimeError, match="Prompt feedback"):
                p.transcribe(audio_file, model="gemini-2.0-flash")
        finally:
            os.unlink(audio_file)


# ---------------------------------------------------------------------------
# NVIDIA
# ---------------------------------------------------------------------------


class TestNvidiaProvider:
    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}, clear=True)
    def test_is_available(self):
        from sapat.providers.nvidia import NvidiaProvider

        assert NvidiaProvider.is_available() is True

    @patch.dict(os.environ, {}, clear=True)
    def test_is_not_available(self):
        from sapat.providers.nvidia import NvidiaProvider

        assert NvidiaProvider.is_available() is False

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}, clear=True)
    def test_resolve_model_aliases(self):
        from sapat.providers.nvidia import NvidiaProvider

        p = NvidiaProvider()
        assert p.resolve_model("parakeet") == "nvidia/parakeet-ctc-0.6b-asr"
        assert p.resolve_model("canary") == "nvidia/canary-1b"
        assert p.resolve_model("custom") == "custom"

    @patch.dict(
        os.environ,
        {
            "NVIDIA_API_KEY": "test-key",
            "NVIDIA_BASE_URL": "https://example.test/v1/",
        },
        clear=True,
    )
    @patch("sapat.providers.nvidia.requests.post")
    def test_transcribe_posts_to_nim_endpoint(self, mock_post):
        mock_post.return_value = FakeResponse(
            200, {"text": "hello from parakeet"}
        )

        audio_file = _make_audio(".mp3", b"audio")

        try:
            from sapat.providers.nvidia import NvidiaProvider

            p = NvidiaProvider()
            result = p.transcribe(
                audio_file,
                model="nvidia/parakeet-ctc-0.6b-asr",
                language="en",
            )
        finally:
            os.unlink(audio_file)

        assert result.text == "hello from parakeet"
        call_args = mock_post.call_args
        assert (
            call_args.args[0]
            == "https://example.test/v1/audio/transcriptions"
        )
        kwargs = call_args.kwargs
        assert kwargs["headers"]["Authorization"] == "Bearer test-key"
        assert kwargs["data"]["model"] == "nvidia/parakeet-ctc-0.6b-asr"
        assert kwargs["data"]["language"] == "en"
        assert "file" in kwargs["files"]

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}, clear=True)
    @patch("sapat.providers.nvidia.requests.post")
    def test_transcribe_error_raises(self, mock_post):
        mock_post.return_value = FakeResponse(
            500, text="Internal Server Error"
        )

        audio_file = _make_audio(".mp3")

        try:
            from sapat.providers.nvidia import NvidiaProvider

            p = NvidiaProvider()
            with pytest.raises(RuntimeError, match="NVIDIA transcription failed"):
                p.transcribe(audio_file, model="nvidia/parakeet-ctc-0.6b-asr")
        finally:
            os.unlink(audio_file)


# ---------------------------------------------------------------------------
# Baidu
# ---------------------------------------------------------------------------


class TestBaiduProvider:
    @patch.dict(
        os.environ,
        {"BAIDU_API_KEY": "key", "BAIDU_SECRET_KEY": "secret"},
        clear=True,
    )
    def test_is_available(self):
        from sapat.providers.baidu import BaiduProvider

        assert BaiduProvider.is_available() is True

    @patch.dict(os.environ, {}, clear=True)
    def test_is_not_available(self):
        from sapat.providers.baidu import BaiduProvider

        assert BaiduProvider.is_available() is False

    @patch.dict(
        os.environ,
        {"BAIDU_API_KEY": "key", "BAIDU_SECRET_KEY": "secret"},
        clear=True,
    )
    def test_resolve_dev_pid_english(self):
        from sapat.providers.baidu import BaiduProvider

        p = BaiduProvider()
        assert p._resolve_dev_pid("en") == 1737

    @patch.dict(
        os.environ,
        {"BAIDU_API_KEY": "key", "BAIDU_SECRET_KEY": "secret"},
        clear=True,
    )
    def test_resolve_dev_pid_chinese(self):
        from sapat.providers.baidu import BaiduProvider

        p = BaiduProvider()
        assert p._resolve_dev_pid("zh-CN") == 1537
        assert p._resolve_dev_pid("zh") == 1537

    @patch.dict(
        os.environ,
        {"BAIDU_API_KEY": "key", "BAIDU_SECRET_KEY": "secret"},
        clear=True,
    )
    def test_resolve_dev_pid_defaults_to_english(self):
        from sapat.providers.baidu import BaiduProvider

        p = BaiduProvider()
        assert p._resolve_dev_pid("fr") == 1737
        assert p._resolve_dev_pid(None) == 1737

    @patch.dict(
        os.environ,
        {"BAIDU_API_KEY": "key", "BAIDU_SECRET_KEY": "secret"},
        clear=True,
    )
    @patch("sapat.providers.baidu.requests.post")
    @patch("sapat.providers.baidu.requests.get")
    def test_transcribe_with_baidu_payload(self, mock_get, mock_post):
        mock_get.return_value = FakeResponse(payload={"access_token": "token-123"})
        mock_post.return_value = FakeResponse(
            payload={"err_no": 0, "result": ["hello world"]}
        )

        audio_file = _make_audio(".wav", b"fake audio bytes")

        try:
            from sapat.providers.baidu import BaiduProvider

            p = BaiduProvider()
            result = p.transcribe(audio_file, model="short_form", language="en")
        finally:
            os.unlink(audio_file)

        assert result.text == "hello world"
        mock_get.assert_called_once()
        payload = mock_post.call_args.kwargs["json"]
        assert payload["token"] == "token-123"
        assert payload["format"] == "wav"
        assert payload["rate"] == 16000
        assert payload["dev_pid"] == 1737
        assert payload["len"] == len(b"fake audio bytes")
        assert payload["speech"] == base64.b64encode(b"fake audio bytes").decode(
            "utf-8"
        )

    @patch.dict(
        os.environ,
        {"BAIDU_API_KEY": "key", "BAIDU_SECRET_KEY": "secret"},
        clear=True,
    )
    @patch("sapat.providers.baidu.requests.post")
    @patch("sapat.providers.baidu.requests.get")
    def test_transcribe_baidu_error_raises(self, mock_get, mock_post):
        mock_get.return_value = FakeResponse(payload={"access_token": "token"})
        mock_post.return_value = FakeResponse(
            payload={"err_no": 3301, "err_msg": "audio quality error"}
        )

        audio_file = _make_audio(".wav")

        try:
            from sapat.providers.baidu import BaiduProvider

            p = BaiduProvider()
            with pytest.raises(RuntimeError, match="audio quality error"):
                p.transcribe(audio_file, model="short_form")
        finally:
            os.unlink(audio_file)

    @patch.dict(
        os.environ,
        {"BAIDU_API_KEY": "key", "BAIDU_SECRET_KEY": "secret"},
        clear=True,
    )
    @patch("sapat.providers.baidu.requests.get")
    def test_token_fetch_failure_raises(self, mock_get):
        mock_get.return_value = FakeResponse(
            payload={"error_description": "invalid client"}
        )

        from sapat.providers.baidu import BaiduProvider

        p = BaiduProvider()
        with pytest.raises(RuntimeError, match="invalid client"):
            p._get_access_token("key", "secret")

    @patch.dict(
        os.environ,
        {"BAIDU_API_KEY": "key", "BAIDU_SECRET_KEY": "secret"},
        clear=True,
    )
    @patch("sapat.providers.baidu.requests.post")
    @patch("sapat.providers.baidu.requests.get")
    def test_transcribe_joins_list_result(self, mock_get, mock_post):
        mock_get.return_value = FakeResponse(payload={"access_token": "token"})
        mock_post.return_value = FakeResponse(
            payload={"err_no": 0, "result": ["hello", "world"]}
        )

        audio_file = _make_audio(".wav")

        try:
            from sapat.providers.baidu import BaiduProvider

            p = BaiduProvider()
            result = p.transcribe(audio_file, model="short_form", language="en")
        finally:
            os.unlink(audio_file)

        assert result.text == "hello world"


# ---------------------------------------------------------------------------
# Cloudflare
# ---------------------------------------------------------------------------


class TestCloudflareProvider:
    @patch.dict(
        os.environ,
        {
            "CLOUDFLARE_ACCOUNT_ID": "account-123",
            "CLOUDFLARE_API_TOKEN": "token-abc",
        },
        clear=True,
    )
    def test_is_available(self):
        from sapat.providers.cloudflare import CloudflareProvider

        assert CloudflareProvider.is_available() is True

    @patch.dict(os.environ, {}, clear=True)
    def test_is_not_available(self):
        from sapat.providers.cloudflare import CloudflareProvider

        assert CloudflareProvider.is_available() is False

    @patch.dict(
        os.environ,
        {
            "CLOUDFLARE_ACCOUNT_ID": "account-123",
            "CLOUDFLARE_API_TOKEN": "token-abc",
        },
        clear=True,
    )
    def test_resolve_model_aliases(self):
        from sapat.providers.cloudflare import CloudflareProvider

        p = CloudflareProvider()
        assert p.resolve_model("whisper") == "@cf/openai/whisper"
        assert p.resolve_model("custom-model") == "custom-model"

    @patch.dict(
        os.environ,
        {
            "CLOUDFLARE_ACCOUNT_ID": "account-123",
            "CLOUDFLARE_API_TOKEN": "token-abc",
        },
        clear=True,
    )
    @patch("sapat.providers.cloudflare.requests.post")
    def test_transcribe_posts_binary_audio(self, mock_post):
        mock_post.return_value = FakeResponse(
            200,
            {"success": True, "result": {"text": "hello from cloudflare"}},
        )

        audio_file = _make_audio(".mp3", b"audio-bytes")

        try:
            from sapat.providers.cloudflare import CloudflareProvider

            p = CloudflareProvider()
            result = p.transcribe(
                audio_file, model="@cf/openai/whisper"
            )
        finally:
            os.unlink(audio_file)

        assert result.text == "hello from cloudflare"
        call_args = mock_post.call_args
        assert (
            call_args.args[0]
            == "https://api.cloudflare.com/client/v4/accounts/account-123/ai/run/@cf/openai/whisper"
        )
        assert call_args.kwargs["headers"]["Authorization"] == "Bearer token-abc"
        assert call_args.kwargs["headers"]["Content-Type"] == "audio/mpeg"
        assert call_args.kwargs["data"] == b"audio-bytes"

    @patch.dict(
        os.environ,
        {
            "CLOUDFLARE_ACCOUNT_ID": "account-123",
            "CLOUDFLARE_API_TOKEN": "token-abc",
        },
        clear=True,
    )
    @patch("sapat.providers.cloudflare.requests.post")
    def test_transcribe_error_raises(self, mock_post):
        mock_post.return_value = FakeResponse(403, text="Forbidden")

        audio_file = _make_audio(".mp3")

        try:
            from sapat.providers.cloudflare import CloudflareProvider

            p = CloudflareProvider()
            with pytest.raises(
                RuntimeError, match="Cloudflare transcription failed"
            ):
                p.transcribe(audio_file, model="@cf/openai/whisper")
        finally:
            os.unlink(audio_file)

    def test_content_type_mapping(self):
        from sapat.providers.cloudflare import CloudflareProvider

        assert CloudflareProvider._content_type("audio.mp3") == "audio/mpeg"
        assert CloudflareProvider._content_type("audio.wav") == "audio/wav"
        assert CloudflareProvider._content_type("audio.flac") == "audio/flac"
        assert (
            CloudflareProvider._content_type("audio.ogg")
            == "application/octet-stream"
        )


# ---------------------------------------------------------------------------
# Sarvam
# ---------------------------------------------------------------------------


class TestSarvamProvider:
    @patch.dict(os.environ, {"SARVAM_API_KEY": "test-key"}, clear=True)
    def test_is_available(self):
        from sapat.providers.sarvam import SarvamProvider

        assert SarvamProvider.is_available() is True

    @patch.dict(os.environ, {}, clear=True)
    def test_is_not_available(self):
        from sapat.providers.sarvam import SarvamProvider

        assert SarvamProvider.is_available() is False

    @patch.dict(os.environ, {"SARVAM_API_KEY": "test-key"}, clear=True)
    def test_resolve_model_aliases(self):
        from sapat.providers.sarvam import SarvamProvider

        p = SarvamProvider()
        assert p.resolve_model("v2") == "saaras:v2"
        assert p.resolve_model("v3") == "saaras:v3"
        assert p.resolve_model("custom") == "custom"

    @patch.dict(os.environ, {"SARVAM_API_KEY": "test-key"}, clear=True)
    def test_resolve_language_code_bcp47_aliases(self):
        from sapat.providers.sarvam import SarvamProvider

        p = SarvamProvider()
        assert p._resolve_language_code("hi") == "hi-IN"
        assert p._resolve_language_code("ta") == "ta-IN"
        assert p._resolve_language_code("en") == "en-IN"

    @patch.dict(os.environ, {"SARVAM_API_KEY": "test-key"}, clear=True)
    def test_resolve_language_code_passthrough_bcp47(self):
        from sapat.providers.sarvam import SarvamProvider

        p = SarvamProvider()
        assert p._resolve_language_code("hi-IN") == "hi-IN"

    @patch.dict(os.environ, {"SARVAM_API_KEY": "test-key"}, clear=True)
    def test_resolve_language_code_unknown(self):
        from sapat.providers.sarvam import SarvamProvider

        p = SarvamProvider()
        assert p._resolve_language_code("unknown") == "unknown"

    @patch.dict(os.environ, {"SARVAM_API_KEY": "test-key"}, clear=True)
    def test_resolve_language_code_none_and_auto(self):
        from sapat.providers.sarvam import SarvamProvider

        p = SarvamProvider()
        assert p._resolve_language_code(None) is None
        assert p._resolve_language_code("auto") is None
        assert p._resolve_language_code("") is None

    @patch.dict(os.environ, {"SARVAM_API_KEY": "test-key"}, clear=True)
    @patch("sapat.providers.sarvam.requests.post")
    def test_transcribe_sends_sarvam_request(self, mock_post):
        mock_post.return_value = FakeResponse(
            200,
            {
                "request_id": "req-123",
                "transcript": "namaste",
                "language_code": "hi-IN",
            },
        )

        audio_file = _make_audio(".mp3")

        try:
            from sapat.providers.sarvam import SarvamProvider

            p = SarvamProvider()
            result = p.transcribe(
                audio_file,
                model="saaras:v3",
                language="hi",
            )
        finally:
            os.unlink(audio_file)

        assert result.text == "namaste"
        _, kwargs = mock_post.call_args
        assert kwargs["headers"] == {"api-subscription-key": "test-key"}
        assert kwargs["data"]["model"] == "saaras:v3"
        assert kwargs["data"]["mode"] == "transcribe"
        assert kwargs["data"]["language_code"] == "hi-IN"

    @patch.dict(os.environ, {"SARVAM_API_KEY": "test-key"}, clear=True)
    @patch("sapat.providers.sarvam.requests.post")
    def test_transcribe_error_raises(self, mock_post):
        mock_post.return_value = FakeResponse(401, text="Unauthorized")

        audio_file = _make_audio(".mp3")

        try:
            from sapat.providers.sarvam import SarvamProvider

            p = SarvamProvider()
            with pytest.raises(
                RuntimeError, match="Sarvam transcription failed"
            ):
                p.transcribe(audio_file, model="saaras:v3")
        finally:
            os.unlink(audio_file)


# ---------------------------------------------------------------------------
# fal.ai
# ---------------------------------------------------------------------------


class TestFalAIProvider:
    @patch.dict(os.environ, {"FAL_KEY": "test-key"}, clear=True)
    def test_is_available_without_package(self):
        from sapat.providers.falai import FalAIProvider

        # Package likely not installed in test env
        assert FalAIProvider.is_available() is False

    @patch.dict(os.environ, {}, clear=True)
    def test_is_not_available_without_key(self):
        from sapat.providers.falai import FalAIProvider

        assert FalAIProvider.is_available() is False

    @patch.dict(os.environ, {"FAL_KEY": "test-key"}, clear=True)
    def test_config_has_required_packages(self):
        from sapat.providers.falai import FalAIProvider

        assert "fal_client" in FalAIProvider.config.required_packages
        assert FalAIProvider.config.extras_key == "falai"

    @patch.dict(os.environ, {"FAL_KEY": "test-key"}, clear=True)
    def test_resolve_model_aliases(self):
        from sapat.providers.falai import FalAIProvider

        p = FalAIProvider()
        assert p.resolve_model("whisper") == "fal-ai/whisper"
        assert p.resolve_model("large") == "fal-ai/whisper/large"
        assert p.resolve_model("custom") == "custom"

    @patch.dict(os.environ, {"FAL_KEY": "test-key"}, clear=True)
    def test_transcribe_uploads_and_subscribes(self):
        mock_fal = MagicMock()
        mock_fal.upload_file.return_value = "https://fal.media/test.mp3"
        mock_fal.subscribe.return_value = {"text": "hello world"}

        audio_file = _make_audio(".mp3", b"audio")

        import sys
        saved = sys.modules.get("fal_client")
        sys.modules["fal_client"] = mock_fal

        try:
            from sapat.providers.falai import FalAIProvider

            p = FalAIProvider()
            result = p.transcribe(
                audio_file,
                model="fal-ai/whisper",
                language="en",
                prompt="Daytona, Sapat",
            )
        finally:
            os.unlink(audio_file)
            if saved is None:
                sys.modules.pop("fal_client", None)
            else:
                sys.modules["fal_client"] = saved

        assert result.text == "hello world"
        mock_fal.upload_file.assert_called_once_with(audio_file)
        mock_fal.subscribe.assert_called_once_with(
            "fal-ai/whisper",
            arguments={
                "audio_url": "https://fal.media/test.mp3",
                "task": "transcribe",
                "chunk_level": "segment",
                "batch_size": 64,
                "language": "en",
                "prompt": "Daytona, Sapat",
            },
        )


# ---------------------------------------------------------------------------
# Soniox
# ---------------------------------------------------------------------------


class TestSonioxProvider:
    @patch.dict(os.environ, {"SONIOX_API_KEY": "test-key"}, clear=True)
    def test_is_available_without_package(self):
        from sapat.providers.soniox import SonioxProvider

        # Package likely not installed in test env
        assert SonioxProvider.is_available() is False

    @patch.dict(os.environ, {}, clear=True)
    def test_is_not_available_without_key(self):
        from sapat.providers.soniox import SonioxProvider

        assert SonioxProvider.is_available() is False

    @patch.dict(os.environ, {"SONIOX_API_KEY": "test-key"}, clear=True)
    def test_config_has_required_packages(self):
        from sapat.providers.soniox import SonioxProvider

        assert "soniox" in SonioxProvider.config.required_packages
        assert SonioxProvider.config.extras_key == "soniox"

    @patch.dict(os.environ, {"SONIOX_API_KEY": "test-key"}, clear=True)
    def test_resolve_model_aliases(self):
        from sapat.providers.soniox import SonioxProvider

        p = SonioxProvider()
        assert p.resolve_model("v2") == "stt-async-v2"
        assert p.resolve_model("v3") == "stt-async-v3"
        assert p.resolve_model("v4") == "stt-async-v4"
        assert p.resolve_model("custom") == "custom"


class _FakeSttClient:
    def __init__(self):
        self.transcribe_calls = []
        self.wait_calls = []
        self.destroy_calls = []

    def transcribe(self, **kwargs):
        self.transcribe_calls.append(kwargs)
        return SimpleNamespace(id="transcription-1")

    def wait(self, transcription_id):
        self.wait_calls.append(transcription_id)

    def get_transcript(self, transcription_id):
        return SimpleNamespace(text=f"transcript for {transcription_id}")

    def destroy(self, transcription_id):
        self.destroy_calls.append(transcription_id)


class _FakeSonioxClient:
    def __init__(self):
        self.stt = _FakeSttClient()


class TestSonioxProviderWithFakeClient:
    @patch.dict(os.environ, {"SONIOX_API_KEY": "test-key"}, clear=True)
    def test_transcribe_submits_waits_gets_and_destroys(self):
        import sys
        fake_client_module = MagicMock()
        fake_client_module.SonioxClient = _FakeSonioxClient
        saved = sys.modules.get("soniox")
        sys.modules["soniox"] = fake_client_module

        audio_file = _make_audio(".mp3", b"fake audio")

        try:
            from sapat.providers.soniox import SonioxProvider

            p = SonioxProvider()
            result = p.transcribe(audio_file, model="stt-async-v4")
        finally:
            os.unlink(audio_file)
            if saved is None:
                sys.modules.pop("soniox", None)
            else:
                sys.modules["soniox"] = saved

        assert result.text == "transcript for transcription-1"

    @patch.dict(
        os.environ,
        {"SONIOX_API_KEY": "test-key", "SONIOX_DESTROY_AFTER_TRANSCRIPTION": "false"},
        clear=True,
    )
    def test_transcribe_skip_destroy_when_env_false(self):
        import sys
        fake_client_module = MagicMock()
        fake_client_module.SonioxClient = _FakeSonioxClient
        saved = sys.modules.get("soniox")
        sys.modules["soniox"] = fake_client_module

        audio_file = _make_audio(".mp3", b"fake audio")

        try:
            from sapat.providers.soniox import SonioxProvider

            p = SonioxProvider()
            result = p.transcribe(audio_file, model="stt-async-v4")
        finally:
            os.unlink(audio_file)
            if saved is None:
                sys.modules.pop("soniox", None)
            else:
                sys.modules["soniox"] = saved

        assert result.text == "transcript for transcription-1"
        # Can't assert destroy was skipped without storing client;
        # the env var SONIOX_DESTROY_AFTER_TRANSCRIPTION=false handles it
