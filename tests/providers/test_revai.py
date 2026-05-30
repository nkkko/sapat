# ABOUTME: Tests for the Rev AI asynchronous transcription provider
# ABOUTME: Verifies upload, polling, transcript fetch, and availability behavior

import json
import os
from unittest.mock import patch

import pytest

from sapat.providers.base import TranscriptionResult


class FakeResponse:
    """Minimal requests.Response stand-in for mocked HTTP calls."""

    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self.payload = payload or {}
        self.text = text

    def json(self):
        return self.payload


@pytest.fixture
def audio_file(tmp_path):
    path = tmp_path / "sample.mp3"
    path.write_bytes(b"fake audio bytes")
    return str(path)


class TestRevAIProvider:
    @patch.dict(
        os.environ,
        {
            "REVAI_ACCESS_TOKEN": "test-token",
            "REVAI_API_BASE_URL": "https://revai.test/speechtotext/v1",
        },
        clear=False,
    )
    @patch("sapat.providers.revai.requests.get")
    @patch("sapat.providers.revai.requests.post")
    def test_transcribe_submits_polls_and_fetches_text(
        self, mock_post, mock_get, audio_file
    ):
        mock_post.return_value = FakeResponse(
            status_code=201,
            payload={"id": "job-123", "status": "in_progress"},
        )
        mock_get.side_effect = [
            FakeResponse(status_code=200, payload={"status": "in_progress"}),
            FakeResponse(status_code=200, payload={"status": "transcribed"}),
            FakeResponse(status_code=200, text="Transcript from Rev AI."),
        ]

        from sapat.providers.revai import RevAIProvider

        provider = RevAIProvider()
        provider.poll_interval = 0.01
        result = provider.transcribe(audio_file, model="async", language="en")

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Transcript from Rev AI."

        mock_post.assert_called_once()
        post_url = mock_post.call_args.args[0]
        post_kwargs = mock_post.call_args.kwargs
        assert post_url == "https://revai.test/speechtotext/v1/jobs"
        assert post_kwargs["headers"]["Authorization"] == "Bearer test-token"
        assert json.loads(post_kwargs["data"]["options"]) == {"language": "en"}
        assert "media" in post_kwargs["files"]
        assert post_kwargs["files"]["media"][2] == "audio/mpeg"

        assert mock_get.call_args_list[0].args[0].endswith("/jobs/job-123")
        assert (
            mock_get.call_args_list[2].args[0]
            == "https://revai.test/speechtotext/v1/jobs/job-123/transcript"
        )
        assert mock_get.call_args_list[2].kwargs["headers"]["Accept"] == "text/plain"

    @patch.dict(os.environ, {"REVAI_ACCESS_TOKEN": "test-token"}, clear=False)
    def test_default_model(self):
        from sapat.providers.revai import RevAIProvider

        assert RevAIProvider.config.default_model == "async"

    @patch.dict(os.environ, {"REVAI_ACCESS_TOKEN": "test-token"}, clear=False)
    def test_job_options_omit_auto_language_and_default_model(self):
        from sapat.providers.revai import RevAIProvider

        provider = RevAIProvider()

        assert provider._job_options("async", "auto") == {}

    @patch.dict(os.environ, {"REVAI_ACCESS_TOKEN": "test-token"}, clear=False)
    def test_job_options_map_model_to_transcriber(self):
        from sapat.providers.revai import RevAIProvider

        provider = RevAIProvider()

        assert provider._job_options("fusion", "en-gb") == {
            "language": "en-gb",
            "transcriber": "fusion",
        }

    @patch.dict(os.environ, {"REVAI_ACCESS_TOKEN": "test-token"}, clear=False)
    def test_available_with_access_token(self):
        from sapat.providers.revai import RevAIProvider

        assert RevAIProvider.is_available() is True

    @patch.dict(os.environ, {}, clear=True)
    def test_not_available_without_access_token(self):
        from sapat.providers.revai import RevAIProvider

        assert RevAIProvider.is_available() is False

    @patch.dict(
        os.environ,
        {
            "REVAI_ACCESS_TOKEN": "test-token",
            "REVAI_API_BASE_URL": "https://revai.test/speechtotext/v1",
        },
        clear=False,
    )
    @patch("sapat.providers.revai.requests.post")
    def test_upload_requires_job_id(self, mock_post, audio_file):
        mock_post.return_value = FakeResponse(status_code=201, payload={})

        from sapat.providers.revai import RevAIProvider

        provider = RevAIProvider()
        with pytest.raises(RuntimeError, match="did not include an id"):
            provider.transcribe(audio_file, model="async")

    @patch.dict(
        os.environ,
        {
            "REVAI_ACCESS_TOKEN": "test-token",
            "REVAI_API_BASE_URL": "https://revai.test/speechtotext/v1",
        },
        clear=False,
    )
    @patch("sapat.providers.revai.requests.get")
    @patch("sapat.providers.revai.requests.post")
    def test_failed_job_raises(self, mock_post, mock_get, audio_file):
        mock_post.return_value = FakeResponse(
            status_code=201, payload={"id": "job-123"}
        )
        mock_get.return_value = FakeResponse(
            status_code=200, payload={"status": "failed"}
        )

        from sapat.providers.revai import RevAIProvider

        provider = RevAIProvider()
        provider.poll_interval = 0.01
        with pytest.raises(RuntimeError, match="failed"):
            provider.transcribe(audio_file, model="async")
