# ABOUTME: Tests for local/offline transcription providers (Group D)
# ABOUTME: Covers Moonshine, Vosk, WhisperX, whisper.cpp, and Picovoice

import os
import subprocess
import sys
import tempfile
import types
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sapat.providers.base import TranscriptionResult


# ---------------------------------------------------------------------------
# Moonshine tests
# ---------------------------------------------------------------------------

class TestMoonshineProvider:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from sapat.providers.moonshine import MoonshineProvider
        self.provider_cls = MoonshineProvider

    def test_config_values(self):
        cfg = self.provider_cls.config
        assert cfg.preferred_format.value == "wav"
        assert cfg.supports_correction is False
        assert cfg.default_model == "moonshine/base"
        assert cfg.max_file_size_mb == float("inf")

    def test_transcript_lines_sorted_chronologically(self):
        p = self.provider_cls.__new__(self.provider_cls)
        transcript = types.SimpleNamespace(
            lines=[
                types.SimpleNamespace(text="later", start_time=4.0),
                types.SimpleNamespace(text="earlier", start_time=1.0),
                types.SimpleNamespace(text=" ", start_time=2.0),
            ]
        )
        result = p._transcript_to_text(transcript)
        assert result == "earlier\nlater"

    def test_transcript_without_lines_returns_str(self):
        p = self.provider_cls.__new__(self.provider_cls)
        result = p._transcript_to_text("raw string")
        assert result == "raw string"

    def test_transcribe_uses_moonshine_voice(self):
        fake_instance = MagicMock()

        class FakeTranscriber:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                FakeTranscriber.last = self

            def transcribe_without_streaming(self, audio_data, sample_rate):
                return types.SimpleNamespace(
                    lines=[
                        types.SimpleNamespace(text="hello", start_time=0.0),
                        types.SimpleNamespace(text="world", start_time=1.0),
                    ]
                )

            def close(self):
                self.closed = True

        fake_module = types.SimpleNamespace(
            Transcriber=FakeTranscriber,
            get_model_for_language=lambda lang: (f"/models/{lang}", "base"),
            load_wav_file=lambda f: ([0.1, -0.1], 16000),
        )

        with patch.dict(sys.modules, {"moonshine_voice": fake_module}):
            with patch.dict(sys.modules, {"moonshine": fake_module}):
                p = self.provider_cls.__new__(self.provider_cls)
                result = p.transcribe("speech.wav", "moonshine/base", language="en")

        assert isinstance(result, TranscriptionResult)
        assert result.text == "hello\nworld"
        assert FakeTranscriber.last.kwargs["model_path"] == "/models/en"
        assert FakeTranscriber.last.closed is True

    def test_missing_dependency_raises_import_error(self):
        with patch.dict(sys.modules, {"moonshine_voice": None, "moonshine": None}):
            p = self.provider_cls.__new__(self.provider_cls)
            with pytest.raises(ImportError, match="sapat\\[moonshine\\]"):
                p.transcribe("speech.wav", "moonshine/base")


# ---------------------------------------------------------------------------
# Vosk tests
# ---------------------------------------------------------------------------

class TestVoskProvider:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from sapat.providers.vosk import VoskProvider
        self.provider_cls = VoskProvider

    def test_config_values(self):
        cfg = self.provider_cls.config
        assert cfg.preferred_format.value == "wav"
        assert cfg.supports_correction is False
        assert cfg.default_model == "vosk-model-small-en-us-0.15"

    def test_extract_text_parses_json(self):
        from sapat.providers.vosk import VoskProvider
        assert VoskProvider._extract_text('{"text": "hello"}') == "hello"
        assert VoskProvider._extract_text("not json") == ""

    def test_validate_rejects_missing_model_path(self):
        p = self.provider_cls.__new__(self.provider_cls)
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            with pytest.raises(ValueError, match="VOSK_MODEL_PATH"):
                p._validate_config(f.name, "")

    def test_validate_rejects_nonexistent_model_dir(self):
        p = self.provider_cls.__new__(self.provider_cls)
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            with pytest.raises(ValueError, match="not a directory"):
                p._validate_config(f.name, "/nonexistent/model/path")

    def test_validate_rejects_missing_audio_file(self):
        p = self.provider_cls.__new__(self.provider_cls)
        with pytest.raises(ValueError, match="does not exist"):
            p._validate_config("/tmp/nonexistent.wav", "/some/model")

    def test_transcribe_processes_audio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()
            wav_path = Path(tmpdir) / "audio.wav"

            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(b"\x00\x00" * 16)

            class FakeRecognizer:
                def __init__(self, model, sr):
                    pass

                def AcceptWaveform(self, data):
                    return True

                def Result(self):
                    return '{"text": "hello"}'

                def FinalResult(self):
                    return '{"text": "world"}'

            fake_vosk = types.SimpleNamespace(
                Model=lambda path: None,
                KaldiRecognizer=FakeRecognizer,
            )

            with patch.dict(sys.modules, {"vosk": fake_vosk}):
                p = self.provider_cls.__new__(self.provider_cls)
                result = p.transcribe(
                    str(wav_path), "model", model_path=str(model_dir)
                )

            assert isinstance(result, TranscriptionResult)
            assert result.text == "hello world"


# ---------------------------------------------------------------------------
# WhisperX tests
# ---------------------------------------------------------------------------

class TestWhisperXProvider:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from sapat.providers.whisperx import WhisperXProvider
        self.provider_cls = WhisperXProvider

    def test_config_values(self):
        cfg = self.provider_cls.config
        assert cfg.preferred_format.value == "wav"
        assert cfg.supports_correction is False
        assert cfg.default_model == "small"

    def test_is_truthy(self):
        from sapat.providers.whisperx import WhisperXProvider
        assert WhisperXProvider._is_truthy("true") is True
        assert WhisperXProvider._is_truthy("1") is True
        assert WhisperXProvider._is_truthy("false") is False
        assert WhisperXProvider._is_truthy(None) is False

    def test_validate_rejects_empty_binary(self):
        p = self.provider_cls.__new__(self.provider_cls)
        with pytest.raises(ValueError, match="WHISPERX_BINARY"):
            p._validate("/tmp/audio.wav", [])

    def test_validate_rejects_missing_binary(self):
        p = self.provider_cls.__new__(self.provider_cls)
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            with pytest.raises(ValueError, match="WhisperX executable"):
                p._validate(f.name, ["/nonexistent/whisperx"])

    def test_validate_rejects_missing_audio_file(self):
        p = self.provider_cls.__new__(self.provider_cls)
        with tempfile.NamedTemporaryFile(suffix=".sh") as binary_f:
            with pytest.raises(ValueError, match="does not exist"):
                p._validate("/nonexistent/audio.wav", [binary_f.name])

    def test_read_txt_output_finds_stem_match(self):
        from sapat.providers.whisperx import WhisperXProvider
        with tempfile.TemporaryDirectory() as tmpdir:
            txt = Path(tmpdir) / "sample.txt"
            txt.write_text("hello whisperx", encoding="utf-8")
            result = WhisperXProvider._read_txt_output(tmpdir, "/path/to/sample.wav")
            assert result == "hello whisperx"

    def test_read_txt_output_falls_back_to_first_txt(self):
        from sapat.providers.whisperx import WhisperXProvider
        with tempfile.TemporaryDirectory() as tmpdir:
            txt = Path(tmpdir) / "other.txt"
            txt.write_text("fallback", encoding="utf-8")
            result = WhisperXProvider._read_txt_output(tmpdir, "/path/to/sample.wav")
            assert result == "fallback"

    def test_read_txt_output_raises_when_no_output(self):
        from sapat.providers.whisperx import WhisperXProvider
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(RuntimeError, match="did not produce"):
                WhisperXProvider._read_txt_output(tmpdir, "/path/to/sample.wav")

    def test_transcribe_builds_command_and_reads_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio = Path(tmpdir) / "sample.wav"
            binary = Path(tmpdir) / "whisperx"
            audio.write_bytes(b"audio")
            binary.write_text("#!/bin/sh\n", encoding="utf-8")

            def fake_run(command, **kwargs):
                output_dir = command[command.index("--output_dir") + 1]
                Path(output_dir, "sample.txt").write_text(
                    "aligned text", encoding="utf-8"
                )
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            env = {
                "WHISPERX_BINARY": str(binary),
                "WHISPERX_MODEL": "small",
                "WHISPERX_DEVICE": "cpu",
                "WHISPERX_COMPUTE_TYPE": "int8",
                "WHISPERX_BATCH_SIZE": "2",
                "WHISPERX_HF_TOKEN": "hf_test",
                "WHISPERX_DIARIZE": "true",
                "WHISPERX_MIN_SPEAKERS": "1",
                "WHISPERX_MAX_SPEAKERS": "3",
            }

            with patch.dict(os.environ, env, clear=True):
                p = self.provider_cls.__new__(self.provider_cls)
                with patch(
                    "sapat.providers.whisperx.subprocess.run",
                    side_effect=fake_run,
                ) as mock_run:
                    result = p.transcribe(
                        str(audio), "small", language="en", prompt="test prompt"
                    )

            assert isinstance(result, TranscriptionResult)
            assert result.text == "aligned text"

            cmd = mock_run.call_args.args[0]
            assert cmd[0] == str(binary)
            assert "--language" in cmd
            assert "en" in cmd
            assert "--initial_prompt" in cmd
            assert "test prompt" in cmd
            assert "--hf_token" in cmd
            assert "--diarize" in cmd


# ---------------------------------------------------------------------------
# whisper.cpp tests
# ---------------------------------------------------------------------------

class TestWhisperCppProvider:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from sapat.providers.whisper_cpp import WhisperCppProvider
        self.provider_cls = WhisperCppProvider

    def test_config_values(self):
        cfg = self.provider_cls.config
        assert cfg.preferred_format.value == "wav"
        assert cfg.supports_correction is False
        assert cfg.default_model == "base"

    def test_clean_output_removes_timestamps_and_runtime_lines(self):
        from sapat.providers.whisper_cpp import WhisperCppProvider
        output = (
            "whisper_init_from_file: loading model\n"
            "[00:00:00.000 --> 00:00:01.000] Hello world.\n"
            "main: processing done\n"
            "Plain line\n"
        )
        result = WhisperCppProvider._clean_output(output)
        assert result == "Hello world.\nPlain line"

    def test_validate_rejects_missing_model_path(self):
        p = self.provider_cls.__new__(self.provider_cls)
        with pytest.raises(ValueError, match="WHISPER_CPP_MODEL_PATH"):
            p._validate("/tmp/audio.wav", "whisper-cli", None)

    def test_validate_rejects_nonexistent_model_file(self):
        p = self.provider_cls.__new__(self.provider_cls)
        with pytest.raises(ValueError, match="model not found"):
            p._validate("/tmp/audio.wav", "whisper-cli", "/nonexistent/model.bin")

    def test_validate_rejects_missing_binary(self):
        p = self.provider_cls.__new__(self.provider_cls)
        with tempfile.NamedTemporaryFile(suffix=".bin") as model_f, \
             tempfile.NamedTemporaryFile(suffix=".wav") as audio_f:
            with pytest.raises(ValueError, match="binary not found"):
                p._validate(audio_f.name, "/nonexistent/binary", model_f.name)

    def test_validate_rejects_missing_audio_file(self):
        p = self.provider_cls.__new__(self.provider_cls)
        with tempfile.NamedTemporaryFile(suffix=".bin") as model_f, \
             tempfile.NamedTemporaryFile(suffix=".sh") as binary_f:
            with pytest.raises(ValueError, match="Audio file not found"):
                p._validate(
                    "/nonexistent/audio.wav", binary_f.name, model_f.name
                )

    def test_transcribe_builds_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = Path(tmpdir) / "ggml-base.en.bin"
            audio = Path(tmpdir) / "sample.wav"
            binary = Path(tmpdir) / "whisper-cli"
            model.write_bytes(b"model")
            audio.write_bytes(b"audio")
            binary.write_text("#!/bin/sh\n", encoding="utf-8")

            completed = subprocess.CompletedProcess(
                args=[], returncode=0,
                stdout="[00:00:00.000 --> 00:00:01.000] Local transcript",
                stderr="",
            )

            env = {
                "WHISPER_CPP_BINARY": str(binary),
                "WHISPER_CPP_MODEL_PATH": str(model),
                "WHISPER_CPP_THREADS": "4",
                "WHISPER_CPP_EXTRA_ARGS": "--no-prints",
            }

            with patch.dict(os.environ, env, clear=True):
                p = self.provider_cls.__new__(self.provider_cls)
                with patch(
                    "sapat.providers.whisper_cpp.subprocess.run",
                    return_value=completed,
                ) as mock_run:
                    result = p.transcribe(
                        str(audio), "base", language="en", prompt="test prompt"
                    )

            assert isinstance(result, TranscriptionResult)
            assert result.text == "Local transcript"

            cmd = mock_run.call_args.args[0]
            assert cmd[0] == str(binary)
            assert "-m" in cmd
            assert str(model) in cmd
            assert "-f" in cmd
            assert str(audio) in cmd
            assert "-nt" in cmd
            assert "-l" in cmd
            assert "en" in cmd
            assert "--prompt" in cmd
            assert "test prompt" in cmd
            assert "-t" in cmd
            assert "4" in cmd
            assert "--no-prints" in cmd


# ---------------------------------------------------------------------------
# Picovoice tests
# ---------------------------------------------------------------------------

class TestPicovoiceProvider:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from sapat.providers.picovoice import PicovoiceProvider
        self.provider_cls = PicovoiceProvider

    def test_config_values(self):
        cfg = self.provider_cls.config
        assert cfg.preferred_format.value == "wav"
        assert cfg.supports_correction is False
        assert "PICOVOICE_ACCESS_KEY" in cfg.required_env_vars

    def test_transcribe_with_factory(self):
        class FakeLeopard:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def process_file(self, audio_file):
                return "hello from leopard", []

            def delete(self):
                self.deleted = True

        fake_engine = FakeLeopard()

        def factory(**kwargs):
            fake_engine.kwargs = kwargs
            return fake_engine

        with patch.dict(os.environ, {"PICOVOICE_ACCESS_KEY": "test-key"}, clear=True):
            p = self.provider_cls.__new__(self.provider_cls)
            p._get_leopard_factory = lambda: factory

            with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                result = p.transcribe(f.name, "leopard")

        assert isinstance(result, TranscriptionResult)
        assert result.text == "hello from leopard"
        assert fake_engine.deleted is True
        assert fake_engine.kwargs["access_key"] == "test-key"

    def test_missing_access_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            p = self.provider_cls.__new__(self.provider_cls)
            with pytest.raises(ValueError, match="PICOVOICE_ACCESS_KEY"):
                p.transcribe("/tmp/audio.wav", "leopard")

    def test_missing_audio_file_raises(self):
        with patch.dict(os.environ, {"PICOVOICE_ACCESS_KEY": "key"}, clear=True):
            p = self.provider_cls.__new__(self.provider_cls)
            with pytest.raises(ValueError, match="does not exist"):
                p.transcribe("/tmp/nonexistent.wav", "leopard")

    def test_bool_from_env(self):
        from sapat.providers.picovoice import PicovoiceProvider
        with patch.dict(os.environ, {"TEST_VAR": "true"}, clear=False):
            assert PicovoiceProvider._bool_from_env("TEST_VAR") is True
        with patch.dict(os.environ, {"TEST_VAR": "false"}, clear=False):
            assert PicovoiceProvider._bool_from_env("TEST_VAR") is False
        assert PicovoiceProvider._bool_from_env("MISSING_VAR", default=True) is True
