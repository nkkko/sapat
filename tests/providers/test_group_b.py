# ABOUTME: Mock-based tests for all 6 transcription providers in group B
# ABOUTME: Tests verify each provider sends correct auth, URL, and payload

import json
import os
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

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
# 1. Symbl.ai
# ===========================================================================


class TestSymblProvider:
    @patch.dict(
        os.environ,
        {
            "SYMBL_ACCESS_TOKEN": "test-token",
            "SYMBL_API_BASE_URL": "https://symbl.test/v1",
        },
        clear=False,
    )
    @patch("sapat.providers.symbl.requests.get")
    @patch("sapat.providers.symbl.requests.post")
    def test_transcribe_submits_polls_and_fetches(self, mock_post, mock_get, audio_file):
        # _upload: POST to /process/audio
        mock_post.return_value = FakeResponse(
            status_code=201,
            payload={"jobId": "job-123", "conversationId": "conv-456"},
        )
        # _poll: GET job status (in_progress then completed)
        # _fetch_result: GET conversation messages
        mock_get.side_effect = [
            FakeResponse(status_code=200, payload={"status": "in_progress"}),
            FakeResponse(status_code=200, payload={"status": "completed"}),
            FakeResponse(
                status_code=200,
                payload={
                    "messages": [
                        {"text": "First sentence."},
                        {"text": "Second sentence."},
                    ],
                },
            ),
        ]

        from sapat.providers.symbl import SymblProvider

        provider = SymblProvider()
        provider.poll_interval = 0.01
        result = provider.transcribe(audio_file, model="default", language="en")

        assert isinstance(result, TranscriptionResult)
        assert result.text == "First sentence.\nSecond sentence."
        mock_post.assert_called_once()
        assert "Authorization" in mock_post.call_args.kwargs["headers"]
        assert mock_post.call_args.kwargs["headers"]["Authorization"] == "Bearer test-token"

    @patch.dict(
        os.environ,
        {
            "SYMBL_ACCESS_TOKEN": "",
            "SYMBL_APP_ID": "app-id",
            "SYMBL_APP_SECRET": "app-secret",
        },
        clear=False,
    )
    @patch("sapat.providers.symbl.requests.post")
    def test_generates_token_from_app_credentials(self, mock_post):
        mock_post.return_value = FakeResponse(
            status_code=200, payload={"accessToken": "generated-token"}
        )

        from sapat.providers.symbl import SymblProvider

        provider = SymblProvider()
        token = provider._get_access_token()

        assert token == "generated-token"

    @patch.dict(
        os.environ,
        {"SYMBL_ACCESS_TOKEN": "test-token"},
        clear=False,
    )
    def test_available_with_token(self):
        from sapat.providers.symbl import SymblProvider

        assert SymblProvider.is_available() is True

    @patch.dict(os.environ, {}, clear=True)
    def test_not_available_without_credentials(self):
        from sapat.providers.symbl import SymblProvider

        assert SymblProvider.is_available() is False

    @patch.dict(
        os.environ,
        {
            "SYMBL_ACCESS_TOKEN": "test-token",
            "SYMBL_API_BASE_URL": "https://symbl.test/v1",
        },
        clear=False,
    )
    @patch("sapat.providers.symbl.requests.get")
    @patch("sapat.providers.symbl.requests.post")
    def test_failed_job_raises(self, mock_post, mock_get, audio_file):
        mock_post.return_value = FakeResponse(
            status_code=201,
            payload={"jobId": "job-123", "conversationId": "conv-456"},
        )
        mock_get.return_value = FakeResponse(
            status_code=200, payload={"status": "failed", "message": "bad audio"}
        )

        from sapat.providers.symbl import SymblProvider

        provider = SymblProvider()
        provider.poll_interval = 0.01
        with pytest.raises(RuntimeError, match="failed"):
            provider.transcribe(audio_file, model="default", language="en")

    @patch.dict(
        os.environ,
        {"SYMBL_ACCESS_TOKEN": "test-token"},
        clear=False,
    )
    def test_default_model(self):
        from sapat.providers.symbl import SymblProvider

        assert SymblProvider.config.default_model == "default"


# ===========================================================================
# 2. Gladia
# ===========================================================================


class TestGladiaProvider:
    @patch.dict(os.environ, {"GLADIA_API_KEY": "test-key"}, clear=False)
    @patch("sapat.providers.gladia.requests.get")
    @patch("sapat.providers.gladia.requests.post")
    def test_transcribe_uploads_creates_job_and_polls(self, mock_post, mock_get, audio_file):
        # Step 1: upload audio -> audio_url
        # Step 2: create transcription job -> result_url
        upload_response = FakeResponse(
            status_code=201,
            payload={"audio_url": "https://api.gladia.io/file/1"},
        )
        job_response = FakeResponse(
            status_code=201,
            payload={
                "id": "job-1",
                "result_url": "https://api.gladia.io/v2/pre-recorded/job-1",
            },
        )
        mock_post.side_effect = [upload_response, job_response]

        # Poll: done
        poll_response = FakeResponse(
            status_code=200,
            payload={
                "status": "done",
                "result": {
                    "transcription": {"full_transcript": "Transcript from Gladia."}
                },
            },
        )
        mock_get.return_value = poll_response

        from sapat.providers.gladia import GladiaProvider

        provider = GladiaProvider()
        provider.poll_interval = 0.01
        result = provider.transcribe(audio_file, model="default", language="en")

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Transcript from Gladia."
        assert mock_post.call_count == 2

    @patch.dict(os.environ, {"GLADIA_API_KEY": "test-key"}, clear=False)
    def test_available_with_key(self):
        from sapat.providers.gladia import GladiaProvider

        assert GladiaProvider.is_available() is True

    @patch.dict(os.environ, {}, clear=True)
    def test_not_available_without_key(self):
        from sapat.providers.gladia import GladiaProvider

        assert GladiaProvider.is_available() is False

    @patch.dict(os.environ, {"GLADIA_API_KEY": "test-key"}, clear=False)
    def test_default_model(self):
        from sapat.providers.gladia import GladiaProvider

        assert GladiaProvider.config.default_model == "default"

    @patch.dict(os.environ, {"GLADIA_API_KEY": "test-key"}, clear=False)
    @patch("sapat.providers.gladia.requests.get")
    @patch("sapat.providers.gladia.requests.post")
    def test_failed_job_raises(self, mock_post, mock_get, audio_file):
        upload_response = FakeResponse(
            status_code=201,
            payload={"audio_url": "https://api.gladia.io/file/1"},
        )
        job_response = FakeResponse(
            status_code=201,
            payload={"id": "job-1"},
        )
        mock_post.side_effect = [upload_response, job_response]
        mock_get.return_value = FakeResponse(
            status_code=200,
            payload={"status": "error"},
        )

        from sapat.providers.gladia import GladiaProvider

        provider = GladiaProvider()
        provider.poll_interval = 0.01
        with pytest.raises(RuntimeError, match="failed"):
            provider.transcribe(audio_file, model="default", language="en")


# ===========================================================================
# 3. Speechmatics
# ===========================================================================


class TestSpeechmaticsProvider:
    @patch.dict(os.environ, {"SPEECHMATICS_API_KEY": "test-key"}, clear=False)
    @patch("sapat.providers.speechmatics.requests.get")
    @patch("sapat.providers.speechmatics.requests.post")
    def test_transcribe_creates_job_and_fetches_text(self, mock_post, mock_get, audio_file):
        create_response = FakeResponse(status_code=201, payload={"id": "job-123"})
        mock_post.return_value = create_response

        # Poll: done
        status_response = FakeResponse(
            status_code=200, payload={"job": {"status": "done"}}
        )
        # Fetch transcript
        transcript_response = FakeResponse(
            status_code=200, text="hello from speechmatics"
        )
        mock_get.side_effect = [status_response, transcript_response]

        from sapat.providers.speechmatics import SpeechmaticsProvider

        provider = SpeechmaticsProvider()
        provider.poll_interval = 0.01
        result = provider.transcribe(audio_file, model="default", language="en")

        assert isinstance(result, TranscriptionResult)
        assert result.text == "hello from speechmatics"
        posted_config = json.loads(mock_post.call_args.kwargs["data"]["config"])
        assert posted_config["type"] == "transcription"
        assert posted_config["transcription_config"]["language"] == "en"

    @patch.dict(os.environ, {"SPEECHMATICS_API_KEY": "test-key"}, clear=False)
    def test_available_with_key(self):
        from sapat.providers.speechmatics import SpeechmaticsProvider

        assert SpeechmaticsProvider.is_available() is True

    @patch.dict(os.environ, {}, clear=True)
    def test_not_available_without_key(self):
        from sapat.providers.speechmatics import SpeechmaticsProvider

        assert SpeechmaticsProvider.is_available() is False

    @patch.dict(os.environ, {"SPEECHMATICS_API_KEY": "test-key"}, clear=False)
    def test_default_model(self):
        from sapat.providers.speechmatics import SpeechmaticsProvider

        assert SpeechmaticsProvider.config.default_model == "default"

    @patch.dict(os.environ, {"SPEECHMATICS_API_KEY": "test-key"}, clear=False)
    @patch("sapat.providers.speechmatics.requests.get")
    @patch("sapat.providers.speechmatics.requests.post")
    def test_rejected_job_raises(self, mock_post, mock_get, audio_file):
        mock_post.return_value = FakeResponse(status_code=201, payload={"id": "job-123"})
        mock_get.return_value = FakeResponse(
            status_code=200, payload={"job": {"status": "rejected"}}
        )

        from sapat.providers.speechmatics import SpeechmaticsProvider

        provider = SpeechmaticsProvider()
        provider.poll_interval = 0.01
        with pytest.raises(RuntimeError, match="failed"):
            provider.transcribe(audio_file, model="default", language="en")


# ===========================================================================
# 4. CAMB.AI
# ===========================================================================


class TestCambAIProvider:
    @patch.dict(
        os.environ,
        {
            "CAMB_API_KEY": "test-key",
            "CAMB_API_BASE_URL": "https://client.camb.test/apis",
        },
        clear=False,
    )
    @patch("sapat.providers.cambai.requests.get")
    @patch("sapat.providers.cambai.requests.post")
    def test_transcribe_uploads_polls_and_fetches_segments(
        self, mock_post, mock_get, audio_file
    ):
        mock_post.return_value = FakeResponse(
            status_code=200,
            payload={"task_id": "task-123"},
        )
        mock_get.side_effect = [
            FakeResponse(status_code=200, payload={"status": "PENDING"}),
            FakeResponse(
                status_code=200,
                payload={"status": "SUCCESS", "run_id": 456},
            ),
            FakeResponse(
                status_code=200,
                payload={
                    "transcript": [
                        {
                            "start": 0.0,
                            "end": 1.4,
                            "speaker": "Speaker 1",
                            "text": "First sentence.",
                        },
                        {
                            "start": 1.5,
                            "end": 3.0,
                            "speaker": "Speaker 2",
                            "text": "Second sentence.",
                        },
                    ]
                },
            ),
        ]

        from sapat.providers.cambai import CambAIProvider

        provider = CambAIProvider()
        provider.poll_interval = 0.01
        result = provider.transcribe(audio_file, model="default", language="en")

        assert isinstance(result, TranscriptionResult)
        assert result.text == "First sentence.\nSecond sentence."
        assert result.segments and len(result.segments) == 2

        mock_post.assert_called_once()
        assert mock_post.call_args.args[0] == "https://client.camb.test/apis/transcribe"
        assert mock_post.call_args.kwargs["headers"]["x-api-key"] == "test-key"
        assert mock_post.call_args.kwargs["data"]["language"] == "en-us"
        assert "media_file" in mock_post.call_args.kwargs["files"]

        assert mock_get.call_args_list[0].args[0] == (
            "https://client.camb.test/apis/transcribe/task-123"
        )
        assert mock_get.call_args_list[-1].args[0] == (
            "https://client.camb.test/apis/transcription-result/456"
        )

    @patch.dict(os.environ, {"CAMB_API_KEY": "test-key"}, clear=False)
    def test_available_with_key(self):
        from sapat.providers.cambai import CambAIProvider

        assert CambAIProvider.is_available() is True

    @patch.dict(os.environ, {}, clear=True)
    def test_not_available_without_key(self):
        from sapat.providers.cambai import CambAIProvider

        assert CambAIProvider.is_available() is False

    @patch.dict(os.environ, {"CAMB_API_KEY": "test-key"}, clear=False)
    def test_default_model(self):
        from sapat.providers.cambai import CambAIProvider

        assert CambAIProvider.config.default_model == "default"

    @patch.dict(os.environ, {"CAMB_API_KEY": "test-key"}, clear=False)
    def test_language_aliases(self):
        from sapat.providers.cambai import CambAIProvider

        provider = CambAIProvider()
        assert provider._normalize_language("en") == "en-us"
        assert provider._normalize_language("pt-br") == "pt-br"
        assert provider._normalize_language(None) == "en-us"

    @patch.dict(os.environ, {"CAMB_API_KEY": "test-key"}, clear=False)
    @patch("sapat.providers.cambai.requests.get")
    @patch("sapat.providers.cambai.requests.post")
    def test_failed_task_raises(self, mock_post, mock_get, audio_file):
        mock_post.return_value = FakeResponse(
            status_code=200,
            payload={"task_id": "task-123"},
        )
        mock_get.return_value = FakeResponse(
            status_code=200,
            payload={"status": "ERROR"},
        )

        from sapat.providers.cambai import CambAIProvider

        provider = CambAIProvider()
        provider.poll_interval = 0.01
        with pytest.raises(RuntimeError, match="failed"):
            provider.transcribe(audio_file, model="default", language="en")


# ===========================================================================
# 5. Yandex SpeechKit
# ===========================================================================


class TestYandexProvider:
    @patch.dict(
        os.environ,
        {
            "YANDEX_API_KEY": "test-api-key",
            "YANDEX_LANGUAGE": "en-US",
            "YANDEX_TOPIC": "general",
            "YANDEX_AUDIO_FORMAT": "oggopus",
        },
        clear=False,
    )
    @patch("sapat.providers.yandex.requests.post")
    @patch("sapat.providers.yandex.subprocess.run")
    def test_transcribe_converts_and_posts(self, mock_run, mock_post, audio_file):
        # Simulate ffmpeg creating the output file
        def fake_run(cmd, **kwargs):
            Path(cmd[-1]).write_bytes(b"fake oggopus")
        mock_run.side_effect = fake_run

        mock_post.return_value = FakeResponse(
            status_code=200, payload={"result": "hello world"}
        )

        from sapat.providers.yandex import YandexProvider

        provider = YandexProvider()
        result = provider.transcribe(audio_file, model="general", language="en")

        assert isinstance(result, TranscriptionResult)
        assert result.text == "hello world"
        mock_run.assert_called_once()
        _, post_kwargs = mock_post.call_args
        assert post_kwargs["headers"]["Authorization"] == "Api-Key test-api-key"
        assert post_kwargs["params"]["lang"] == "en-US"
        assert post_kwargs["params"]["format"] == "oggopus"

    @patch.dict(
        os.environ,
        {
            "YANDEX_API_KEY": "",
            "YANDEX_IAM_TOKEN": "iam-token",
            "YANDEX_FOLDER_ID": "folder-123",
            "YANDEX_LANGUAGE": "en-US",
            "YANDEX_TOPIC": "general",
            "YANDEX_AUDIO_FORMAT": "oggopus",
        },
        clear=False,
    )
    @patch("sapat.providers.yandex.requests.post")
    @patch("sapat.providers.yandex.subprocess.run")
    def test_iam_token_auth_with_folder_id(self, mock_run, mock_post, audio_file):
        def fake_run(cmd, **kwargs):
            Path(cmd[-1]).write_bytes(b"fake oggopus")
        mock_run.side_effect = fake_run

        mock_post.return_value = FakeResponse(
            status_code=200, payload={"result": "token auth"}
        )

        from sapat.providers.yandex import YandexProvider

        provider = YandexProvider()
        result = provider.transcribe(audio_file, model="general")

        assert result.text == "token auth"
        _, post_kwargs = mock_post.call_args
        assert post_kwargs["headers"]["Authorization"] == "Bearer iam-token"
        assert post_kwargs["params"]["folderId"] == "folder-123"

    @patch.dict(os.environ, {"YANDEX_API_KEY": "test-key"}, clear=False)
    def test_available_with_api_key(self):
        from sapat.providers.yandex import YandexProvider

        assert YandexProvider.is_available() is True

    @patch.dict(
        os.environ, {"YANDEX_API_KEY": "", "YANDEX_IAM_TOKEN": "tok"}, clear=False
    )
    def test_available_with_iam_token(self):
        from sapat.providers.yandex import YandexProvider

        assert YandexProvider.is_available() is True

    @patch.dict(os.environ, {}, clear=True)
    def test_not_available_without_credentials(self):
        from sapat.providers.yandex import YandexProvider

        assert YandexProvider.is_available() is False

    @patch.dict(os.environ, {"YANDEX_API_KEY": "test-key"}, clear=False)
    def test_default_model(self):
        from sapat.providers.yandex import YandexProvider

        assert YandexProvider.config.default_model == "general"


# ===========================================================================
# 5. Oracle Cloud AI Speech
# ===========================================================================


def _fake_oci(object_client, speech_client):
    """Build a fake oci module for testing Oracle provider."""
    def record_factory(name):
        def factory(**kwargs):
            return types.SimpleNamespace(_model_name=name, **kwargs)
        return factory

    models = types.SimpleNamespace(
        CreateTranscriptionJobDetails=record_factory("CreateTranscriptionJobDetails"),
        ObjectListInlineInputLocation=record_factory("ObjectListInlineInputLocation"),
        ObjectLocation=record_factory("ObjectLocation"),
        OutputLocation=record_factory("OutputLocation"),
        TranscriptionModelDetails=record_factory("TranscriptionModelDetails"),
    )
    return types.SimpleNamespace(
        config=types.SimpleNamespace(
            from_file=Mock(return_value={"region": "us-ashburn-1"})
        ),
        object_storage=types.SimpleNamespace(
            ObjectStorageClient=Mock(return_value=object_client)
        ),
        ai_speech=types.SimpleNamespace(
            AIServiceSpeechClient=Mock(return_value=speech_client),
            models=models,
        ),
    )


class TestOracleProvider:
    @patch.dict(
        os.environ,
        {
            "OCI_COMPARTMENT_ID": "ocid1.compartment.oc1..example",
            "OCI_OBJECT_STORAGE_NAMESPACE": "ns",
            "OCI_SPEECH_INPUT_BUCKET": "in",
            "OCI_SPEECH_OUTPUT_BUCKET": "out",
        },
        clear=False,
    )
    def test_transcribe_uploads_polls_and_reads_output(self, audio_file):
        object_client = Mock()
        speech_client = Mock()
        speech_client.create_transcription_job.return_value.data = types.SimpleNamespace(
            id="job1"
        )
        speech_client.list_transcription_tasks.return_value.data = types.SimpleNamespace(
            items=[
                types.SimpleNamespace(id="task1", lifecycle_state="SUCCEEDED")
            ]
        )
        speech_client.get_transcription_task.return_value.data = types.SimpleNamespace(
            output_location=types.SimpleNamespace(
                namespace_name="ns",
                bucket_name="out",
                object_names=["sapat-transcripts/job1/sample.json"],
            ),
        )
        object_client.get_object.return_value.data.content = json.dumps(
            {"transcriptions": [{"transcription": "oracle transcript"}]}
        ).encode("utf-8")

        fake_oci = _fake_oci(object_client, speech_client)

        from sapat.providers.oracle import OracleProvider

        provider = OracleProvider()
        provider.poll_interval = 0.01
        with patch.object(OracleProvider, "_load_oci", return_value=fake_oci):
            result = provider.transcribe(audio_file, model="default", language="en")

        assert isinstance(result, TranscriptionResult)
        assert result.text == "oracle transcript"
        object_client.put_object.assert_called_once()
        speech_client.create_transcription_job.assert_called_once()

    @patch.dict(os.environ, {}, clear=True)
    def test_not_available_without_compartment_id(self):
        from sapat.providers.oracle import OracleProvider

        assert OracleProvider.is_available() is False

    @patch.dict(
        os.environ,
        {
            "OCI_COMPARTMENT_ID": "ocid1.compartment.oc1..example",
            "OCI_OBJECT_STORAGE_NAMESPACE": "ns",
            "OCI_SPEECH_INPUT_BUCKET": "in",
            "OCI_SPEECH_OUTPUT_BUCKET": "out",
        },
        clear=False,
    )
    def test_default_model(self):
        from sapat.providers.oracle import OracleProvider

        assert OracleProvider.config.default_model == "default"

    @patch.dict(
        os.environ,
        {
            "OCI_COMPARTMENT_ID": "ocid1.compartment.oc1..example",
            "OCI_OBJECT_STORAGE_NAMESPACE": "ns",
            "OCI_SPEECH_INPUT_BUCKET": "in",
            "OCI_SPEECH_OUTPUT_BUCKET": "out",
        },
        clear=False,
    )
    def test_oracle_model_requires_locale(self):
        from sapat.providers.oracle import OracleProvider

        provider = OracleProvider()
        assert provider._normalize_language("es") == "es-ES"
        with pytest.raises(ValueError, match="requires an explicit locale"):
            provider._normalize_language("auto")

    def test_extract_text_from_json(self):
        from sapat.providers.oracle import OracleProvider

        raw = json.dumps({"transcriptions": [{"transcription": "hello"}, {"transcription": "world"}]})
        assert OracleProvider._extract_text(raw) == "hello\nworld"

    def test_extract_text_from_plain_text(self):
        from sapat.providers.oracle import OracleProvider

        assert OracleProvider._extract_text("plain text") == "plain text"


# ===========================================================================
# 6. Replicate
# ===========================================================================


class TestReplicateProvider:
    @patch.dict(
        os.environ,
        {
            "REPLICATE_API_TOKEN": "test-token",
            "REPLICATE_MODEL": "openai/whisper",
        },
        clear=False,
    )
    def test_transcribe_sends_correct_model_and_input(self, audio_file):
        mock_replicate = MagicMock()
        mock_replicate.Client.return_value.run.return_value = {
            "transcription": "hello from replicate"
        }

        from sapat.providers.replicate import ReplicateProvider

        provider = ReplicateProvider()
        with patch.dict("sys.modules", {"replicate": mock_replicate}):
            result = provider.transcribe(audio_file, model="openai/whisper")

        assert isinstance(result, TranscriptionResult)
        assert result.text == "hello from replicate"
        mock_replicate.Client.assert_called_once_with(api_token="test-token")
        mock_replicate.Client.return_value.run.assert_called_once()
        call_args = mock_replicate.Client.return_value.run.call_args
        assert call_args.args[0] == "openai/whisper"

    @patch.dict(
        os.environ,
        {
            "REPLICATE_API_TOKEN": "test-token",
            "REPLICATE_MODEL": "openai/whisper",
        },
        clear=False,
    )
    def test_translate_flag_passed_through(self, audio_file):
        mock_replicate = MagicMock()
        mock_replicate.Client.return_value.run.return_value = {"text": "translated text"}

        from sapat.providers.replicate import ReplicateProvider

        provider = ReplicateProvider()
        with patch.dict("sys.modules", {"replicate": mock_replicate}):
            provider.transcribe(audio_file, model="openai/whisper", translate=True)

        call_kwargs = mock_replicate.Client.return_value.run.call_args.kwargs
        assert call_kwargs["input"]["translate"] is True

    @patch.dict(os.environ, {"REPLICATE_API_TOKEN": "test-token"}, clear=False)
    def test_available_with_token_and_package(self):
        from sapat.providers.replicate import ReplicateProvider

        with patch.dict("sys.modules", {"replicate": MagicMock()}):
            assert ReplicateProvider.is_available() is True

    @patch.dict(os.environ, {}, clear=True)
    def test_not_available_without_token(self):
        from sapat.providers.replicate import ReplicateProvider

        assert ReplicateProvider.is_available() is False

    @patch.dict(os.environ, {"REPLICATE_API_TOKEN": "test-token"}, clear=False)
    def test_default_model(self):
        from sapat.providers.replicate import ReplicateProvider

        assert ReplicateProvider.config.default_model == "openai/whisper"

    def test_extract_text_from_string(self):
        from sapat.providers.replicate import ReplicateProvider

        provider = ReplicateProvider.__new__(ReplicateProvider)
        assert provider._extract_transcription_text("plain string") == "plain string"

    def test_extract_text_from_segments(self):
        from sapat.providers.replicate import ReplicateProvider

        provider = ReplicateProvider.__new__(ReplicateProvider)
        output = {"segments": [{"text": "hello"}, {"text": " world"}]}
        assert provider._extract_transcription_text(output) == "hello world"

    def test_extract_text_from_list(self):
        from sapat.providers.replicate import ReplicateProvider

        provider = ReplicateProvider.__new__(ReplicateProvider)
        assert provider._extract_transcription_text(["part1", "part2"]) == "part1part2"

    def test_unsupported_output_raises(self):
        from sapat.providers.replicate import ReplicateProvider

        provider = ReplicateProvider.__new__(ReplicateProvider)
        with pytest.raises(ValueError, match="Unsupported Replicate"):
            provider._extract_transcription_text({"segments": [{"timestamp": [0, 1]}]})

    @patch.dict(os.environ, {"REPLICATE_API_TOKEN": "test-token"}, clear=False)
    def test_resolve_model_alias(self):
        from sapat.providers.replicate import ReplicateProvider

        provider = ReplicateProvider()
        assert provider.resolve_model("whisper") == "openai/whisper"
        assert provider.resolve_model("custom/model") == "custom/model"
