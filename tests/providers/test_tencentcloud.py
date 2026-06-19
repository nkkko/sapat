# ABOUTME: Mock-based tests for Tencent Cloud ASR provider
# ABOUTME: Verifies signed request payloads, response parsing, and errors

import base64
import json
from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def audio_file(tmp_path):
    path = tmp_path / "sample.mp3"
    path.write_bytes(b"fake audio bytes")
    return str(path)


@pytest.fixture
def tencent_env():
    return {
        "TENCENTCLOUD_SECRET_ID": "test-secret-id",
        "TENCENTCLOUD_SECRET_KEY": "test-secret-key",
        "TENCENTCLOUD_REGION": "ap-guangzhou",
        "TENCENTCLOUD_TIMESTAMP": "1700000000",
    }


class TestTencentCloudProvider:
    def test_config_values(self):
        from sapat.providers.tencentcloud import TencentCloudProvider

        cfg = TencentCloudProvider.config
        assert cfg.required_env_vars == [
            "TENCENTCLOUD_SECRET_ID",
            "TENCENTCLOUD_SECRET_KEY",
        ]
        assert cfg.preferred_format.value == "mp3"
        assert cfg.max_file_size_mb == 3.0
        assert cfg.default_model == "16k_zh"

    def test_is_available_when_credentials_are_set(self, tencent_env):
        from sapat.providers.tencentcloud import TencentCloudProvider

        with patch.dict("os.environ", tencent_env, clear=True):
            assert TencentCloudProvider.is_available() is True

    @patch("sapat.providers.tencentcloud.requests.post")
    def test_transcribe_posts_signed_sentence_recognition_request(
        self, mock_post, audio_file, tencent_env
    ):
        from sapat.providers.tencentcloud import TencentCloudProvider

        mock_post.return_value = Mock(
            status_code=200,
            json=lambda: {
                "Response": {
                    "Result": "hello from tencent",
                    "AudioDuration": 2500,
                    "WordList": [{"Word": "hello", "StartTime": 0, "EndTime": 500}],
                    "RequestId": "req-1",
                }
            },
            text="ok",
        )

        with patch.dict("os.environ", tencent_env, clear=True):
            provider = TencentCloudProvider()
            result = provider.transcribe(
                audio_file,
                "english",
                language="en",
                word_info=1,
                hotword_list="Sapat|10,Daytona|8",
            )

        assert result.text == "hello from tencent"
        assert result.duration == 2.5
        assert result.segments == [{"Word": "hello", "StartTime": 0, "EndTime": 500}]

        url = mock_post.call_args.args[0]
        headers = mock_post.call_args.kwargs["headers"]
        payload = json.loads(mock_post.call_args.kwargs["data"])

        assert url == "https://asr.tencentcloudapi.com"
        assert headers["X-TC-Action"] == "SentenceRecognition"
        assert headers["X-TC-Version"] == "2019-06-14"
        assert headers["X-TC-Region"] == "ap-guangzhou"
        assert headers["X-TC-Timestamp"] == "1700000000"
        assert headers["Authorization"].startswith(
            "TC3-HMAC-SHA256 Credential=test-secret-id/"
        )
        assert payload["EngSerViceType"] == "16k_en"
        assert payload["SourceType"] == 1
        assert payload["VoiceFormat"] == "mp3"
        assert payload["ProjectId"] == 0
        assert payload["SubServiceType"] == 2
        assert payload["Data"] == base64.b64encode(b"fake audio bytes").decode("utf-8")
        assert payload["DataLen"] == len(b"fake audio bytes")
        assert payload["WordInfo"] == 1
        assert payload["HotwordList"] == "Sapat|10,Daytona|8"

    @patch("sapat.providers.tencentcloud.requests.post")
    def test_transcribe_raises_on_tencent_error(
        self, mock_post, audio_file, tencent_env
    ):
        from sapat.providers.tencentcloud import TencentCloudProvider

        mock_post.return_value = Mock(
            status_code=200,
            json=lambda: {
                "Response": {
                    "Error": {
                        "Code": "AuthFailure.SignatureFailure",
                        "Message": "bad signature",
                    },
                    "RequestId": "req-2",
                }
            },
            text="bad signature",
        )

        with patch.dict("os.environ", tencent_env, clear=True):
            provider = TencentCloudProvider()
            with pytest.raises(RuntimeError, match="bad signature"):
                provider.transcribe(audio_file, "default")

    def test_resolve_model_aliases_and_language_default(self):
        from sapat.providers.tencentcloud import TencentCloudProvider

        provider = TencentCloudProvider.__new__(TencentCloudProvider)
        assert provider.resolve_model("english") == "16k_en"
        assert provider.resolve_model("cantonese") == "16k_yue"
        assert provider._model_for_language("ja") == "16k_ja"
        assert provider._model_for_language("unknown") == "16k_zh"
