# ABOUTME: Tests for the local SpeechBrain transcription provider
# ABOUTME: Uses mocked inference classes so no models or dependencies are downloaded

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from sapat.providers.base import TranscriptionResult
from sapat.providers.speechbrain import SpeechBrainProvider


class TestSpeechBrainProvider:
    def test_config_values(self):
        config = SpeechBrainProvider.config

        assert config.required_env_vars == []
        assert config.required_packages == ["speechbrain"]
        assert config.extras_key == "speechbrain"
        assert config.preferred_format.value == "wav"
        assert config.default_model == "speechbrain/asr-crdnn-rnnlm-librispeech"

    def test_is_available_with_mock_dependency(self):
        fake_speechbrain = types.ModuleType("speechbrain")

        with patch.dict(sys.modules, {"speechbrain": fake_speechbrain}):
            assert SpeechBrainProvider.is_available() is True

    def test_env_model_overrides_default_model(self, tmp_path):
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"wav")
        recognizer = MagicMock()
        recognizer.transcribe_file.return_value = "hello"
        asr_class = MagicMock()
        asr_class.from_hparams.return_value = recognizer

        env = {"SPEECHBRAIN_MODEL": "org/model-from-env"}
        with patch.dict("os.environ", env, clear=True):
            with patch.object(
                SpeechBrainProvider, "_get_asr_class", return_value=asr_class
            ):
                SpeechBrainProvider().transcribe(
                    str(audio), model=SpeechBrainProvider.config.default_model
                )

        assert asr_class.from_hparams.call_args.kwargs["source"] == "org/model-from-env"

    def test_explicit_non_default_model_beats_env_override(self, tmp_path):
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"wav")
        recognizer = MagicMock()
        recognizer.transcribe_file.return_value = "hello"
        asr_class = MagicMock()
        asr_class.from_hparams.return_value = recognizer

        with patch.dict(
            "os.environ", {"SPEECHBRAIN_MODEL": "org/env-model"}, clear=True
        ):
            with patch.object(
                SpeechBrainProvider, "_get_asr_class", return_value=asr_class
            ):
                SpeechBrainProvider().transcribe(str(audio), model="org/explicit-model")

        assert asr_class.from_hparams.call_args.kwargs["source"] == "org/explicit-model"

    def test_model_argument_used_without_env_override(self, tmp_path):
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"wav")
        recognizer = MagicMock()
        recognizer.transcribe_file.return_value = "hello"
        asr_class = MagicMock()
        asr_class.from_hparams.return_value = recognizer

        with patch.dict("os.environ", {}, clear=True):
            with patch.object(
                SpeechBrainProvider, "_get_asr_class", return_value=asr_class
            ):
                SpeechBrainProvider().transcribe(str(audio), model="org/explicit-model")

        assert asr_class.from_hparams.call_args.kwargs["source"] == "org/explicit-model"

    def test_from_hparams_receives_savedir_and_device(self, tmp_path):
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"wav")
        recognizer = MagicMock()
        recognizer.transcribe_file.return_value = ("hello world", [1, 2])
        asr_class = MagicMock()
        asr_class.from_hparams.return_value = recognizer

        env = {
            "SPEECHBRAIN_SAVEDIR": str(tmp_path / "models"),
            "SPEECHBRAIN_DEVICE": "cuda:0",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch.object(
                SpeechBrainProvider, "_get_asr_class", return_value=asr_class
            ):
                result = SpeechBrainProvider().transcribe(
                    str(audio), model="speechbrain/test-model"
                )

        asr_class.from_hparams.assert_called_once_with(
            source="speechbrain/test-model",
            savedir=str(tmp_path / "models"),
            run_opts={"device": "cuda:0"},
        )
        recognizer.transcribe_file.assert_called_once_with(str(audio))
        assert isinstance(result, TranscriptionResult)
        assert result.text == "hello world"
        assert result.raw_response == ("hello world", [1, 2])

    def test_from_hparams_omits_unset_savedir_and_defaults_to_cpu(self, tmp_path):
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"wav")
        recognizer = MagicMock()
        recognizer.transcribe_file.return_value = "hello"
        asr_class = MagicMock()
        asr_class.from_hparams.return_value = recognizer

        with patch.dict("os.environ", {}, clear=True):
            with patch.object(
                SpeechBrainProvider, "_get_asr_class", return_value=asr_class
            ):
                SpeechBrainProvider().transcribe(str(audio), model="speechbrain/model")

        asr_class.from_hparams.assert_called_once_with(
            source="speechbrain/model", run_opts={"device": "cpu"}
        )

    @pytest.mark.parametrize(
        ("output", "expected"),
        [
            ("  direct text  ", "direct text"),
            (("tuple text", [1, 2, 3]), "tuple text"),
            (["first", "second"], "first second"),
            ([("first", [1]), ("second", [2])], "first second"),
            ([], ""),
            (None, ""),
        ],
    )
    def test_extracts_common_outputs(self, output, expected):
        assert SpeechBrainProvider._extract_text(output) == expected

    def test_rejects_unsupported_output(self):
        with pytest.raises(ValueError, match="Unsupported SpeechBrain"):
            SpeechBrainProvider._extract_text({"text": "unexpected"})

    def test_missing_audio_fails_before_loading_dependency(self, tmp_path):
        with patch.object(SpeechBrainProvider, "_get_asr_class") as get_asr_class:
            with pytest.raises(ValueError, match="does not exist"):
                SpeechBrainProvider().transcribe(
                    str(tmp_path / "missing.wav"), model="speechbrain/model"
                )

        get_asr_class.assert_not_called()

    def test_transcription_failure_is_contextualized(self, tmp_path):
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"wav")
        recognizer = MagicMock()
        recognizer.transcribe_file.side_effect = OSError("decoder failed")
        asr_class = MagicMock()
        asr_class.from_hparams.return_value = recognizer

        with patch.object(
            SpeechBrainProvider, "_get_asr_class", return_value=asr_class
        ):
            with pytest.raises(
                RuntimeError, match="SpeechBrain transcription failed: decoder failed"
            ):
                SpeechBrainProvider().transcribe(str(audio), model="speechbrain/model")

    def test_lazy_import_uses_maintained_inference_api(self):
        asr_class = object()
        asr_module = types.ModuleType("speechbrain.inference.ASR")
        asr_module.EncoderDecoderASR = asr_class
        inference_module = types.ModuleType("speechbrain.inference")
        inference_module.__path__ = []
        speechbrain_module = types.ModuleType("speechbrain")
        speechbrain_module.__path__ = []

        fake_modules = {
            "speechbrain": speechbrain_module,
            "speechbrain.inference": inference_module,
            "speechbrain.inference.ASR": asr_module,
        }
        with patch.dict(sys.modules, fake_modules):
            assert SpeechBrainProvider._get_asr_class() is asr_class

    def test_missing_dependency_has_install_hint(self):
        with patch.dict(sys.modules, {"speechbrain.inference.ASR": None}):
            with pytest.raises(ImportError, match=r"sapat\[speechbrain\]"):
                SpeechBrainProvider._get_asr_class()
