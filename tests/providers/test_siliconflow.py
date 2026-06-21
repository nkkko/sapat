# ABOUTME: Tests for the SiliconFlow transcription provider
# ABOUTME: Verifies request shape, model aliases, and availability checks

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


class TestSiliconFlowProvider:
    @patch.dict(os.environ, {"SILICONFLOW_API_KEY": "test-key"}, clear=False)
    @patch("sapat.providers.openai_compat.requests.post")
    def test_transcribe_sends_correct_request(self, mock_post, tmp_path):
        audio_file = tmp_path / "sample.mp3"
        audio_file.write_bytes(b"fake audio")
        mock_post.return_value = FakeResponse(payload={"text": "hello siliconflow"})

        from sapat.providers.siliconflow import SiliconFlowProvider

        provider = SiliconFlowProvider()
        result = provider.transcribe(
            str(audio_file),
            model="FunAudioLLM/SenseVoiceSmall",
            language="zh",
            prompt="Product names: Sapat, Daytona",
            temperature=0.1,
        )

        assert isinstance(result, TranscriptionResult)
        assert result.text == "hello siliconflow"

        _, kwargs = mock_post.call_args
        assert mock_post.call_args.args[0] == (
            "https://api.siliconflow.cn/v1/audio/transcriptions"
        )
        assert kwargs["headers"]["Authorization"] == "Bearer test-key"
        assert kwargs["data"]["model"] == "FunAudioLLM/SenseVoiceSmall"
        assert "language" not in kwargs["data"]
        assert "prompt" not in kwargs["data"]
        assert "temperature" not in kwargs["data"]
        assert "file" in kwargs["files"]

    @patch.dict(os.environ, {"SILICONFLOW_API_KEY": "test-key"}, clear=False)
    def test_default_model_and_limits(self):
        from sapat.providers.siliconflow import SiliconFlowProvider

        assert SiliconFlowProvider.config.default_model == "FunAudioLLM/SenseVoiceSmall"
        assert SiliconFlowProvider.config.max_file_size_mb == 50.0
        assert SiliconFlowProvider.config.preferred_format.value == "mp3"

    @patch.dict(os.environ, {"SILICONFLOW_API_KEY": "test-key"}, clear=False)
    def test_resolve_model_aliases(self):
        from sapat.providers.siliconflow import SiliconFlowProvider

        provider = SiliconFlowProvider()
        assert provider.resolve_model("sensevoice") == "FunAudioLLM/SenseVoiceSmall"
        assert (
            provider.resolve_model("sensevoice-small") == "FunAudioLLM/SenseVoiceSmall"
        )
        assert provider.resolve_model("teleai") == "TeleAI/TeleSpeechASR"
        assert provider.resolve_model("custom/model") == "custom/model"

    @patch.dict(os.environ, {}, clear=True)
    def test_not_available_without_key(self):
        from sapat.providers.siliconflow import SiliconFlowProvider

        assert SiliconFlowProvider.is_available() is False

    @patch.dict(os.environ, {"SILICONFLOW_API_KEY": "bad-key"}, clear=False)
    @patch("sapat.providers.openai_compat.requests.post")
    def test_raises_on_api_error(self, mock_post, tmp_path):
        audio_file = tmp_path / "sample.mp3"
        audio_file.write_bytes(b"fake audio")
        mock_post.return_value = FakeResponse(status_code=401, text="unauthorized")

        from sapat.providers.siliconflow import SiliconFlowProvider

        provider = SiliconFlowProvider()
        with pytest.raises(
            RuntimeError, match="siliconflow transcription failed \\(401\\)"
        ):
            provider.transcribe(
                str(audio_file),
                model="FunAudioLLM/SenseVoiceSmall",
            )
