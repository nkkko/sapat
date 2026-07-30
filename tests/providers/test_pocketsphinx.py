# ABOUTME: Tests for the PocketSphinx local transcription provider
# ABOUTME: Uses fake pocketsphinx modules so no model downloads are required

import sys
import types
import wave
from pathlib import Path
from unittest.mock import patch

import pytest

from sapat.providers.base import TranscriptionResult


class TestPocketSphinxProvider:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from sapat.providers.pocketsphinx import PocketSphinxProvider

        self.provider_cls = PocketSphinxProvider

    def test_config_values(self):
        cfg = self.provider_cls.config
        assert cfg.preferred_format.value == "wav"
        assert cfg.supports_correction is False
        assert cfg.default_model == "default"
        assert cfg.max_file_size_mb == float("inf")
        assert cfg.extras_key == "pocketsphinx"

    def test_build_options_uses_custom_paths(self, tmp_path):
        audio = tmp_path / "speech.wav"
        hmm = tmp_path / "hmm"
        lm = tmp_path / "language.lm"
        dictionary = tmp_path / "words.dict"
        kws = tmp_path / "keywords.list"
        audio.write_bytes(b"audio")
        hmm.mkdir()
        lm.write_text("lm", encoding="utf-8")
        dictionary.write_text("dict", encoding="utf-8")
        kws.write_text("hello /1e-20/", encoding="utf-8")

        env = {
            "POCKETSPHINX_HMM": str(hmm),
            "POCKETSPHINX_LM": str(lm),
            "POCKETSPHINX_DICT": str(dictionary),
            "POCKETSPHINX_KWS": str(kws),
            "POCKETSPHINX_KWS_THRESHOLD": "1e-20",
        }
        with patch.dict("os.environ", env, clear=True):
            provider = self.provider_cls.__new__(self.provider_cls)
            options = provider._build_recognizer_options(str(audio), "default")

        assert options["hmm"] == str(hmm)
        assert options["lm"] == str(lm)
        assert options["dict"] == str(dictionary)
        assert options["kws"] == str(kws)
        assert options["kws_threshold"] == 1e-20

    def test_model_argument_can_select_language_model(self, tmp_path):
        audio = tmp_path / "speech.wav"
        lm = tmp_path / "domain.lm"
        audio.write_bytes(b"audio")
        lm.write_text("lm", encoding="utf-8")

        provider = self.provider_cls.__new__(self.provider_cls)
        options = provider._build_recognizer_options(str(audio), str(lm))

        assert options["lm"] == str(lm)

    def test_missing_audio_file_raises(self):
        provider = self.provider_cls.__new__(self.provider_cls)
        with pytest.raises(ValueError, match="Audio file not found"):
            provider._build_recognizer_options("/tmp/not-real.wav", "default")

    def test_missing_custom_path_raises(self, tmp_path):
        audio = tmp_path / "speech.wav"
        audio.write_bytes(b"audio")

        with patch.dict(
            "os.environ", {"POCKETSPHINX_LM": "/tmp/not-real.lm"}, clear=True
        ):
            provider = self.provider_cls.__new__(self.provider_cls)
            with pytest.raises(ValueError, match="lm path not found"):
                provider._build_recognizer_options(str(audio), "default")

    def test_invalid_threshold_raises(self, tmp_path):
        audio = tmp_path / "speech.wav"
        audio.write_bytes(b"audio")

        with patch.dict(
            "os.environ", {"POCKETSPHINX_KWS_THRESHOLD": "nope"}, clear=True
        ):
            provider = self.provider_cls.__new__(self.provider_cls)
            with pytest.raises(ValueError, match="must be a number"):
                provider._build_recognizer_options(str(audio), "default")

    def test_keyword_mode_disables_language_model(self, tmp_path):
        audio = tmp_path / "speech.wav"
        audio.write_bytes(b"audio")

        with patch.dict("os.environ", {"POCKETSPHINX_KEYPHRASE": "hello"}, clear=True):
            provider = self.provider_cls.__new__(self.provider_cls)
            options = provider._build_recognizer_options(str(audio), "default")

        assert options["lm"] is False
        assert options["keyphrase"] == "hello"

    def test_transcribe_collects_phrases(self, tmp_path):
        audio = tmp_path / "speech.wav"
        with wave.open(str(audio), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\x00\x00" * 16000)

        class FakeSegmenter:
            last_sample_rate = None

            def __init__(self, sample_rate):
                self.sample_rate = sample_rate
                FakeSegmenter.last_sample_rate = sample_rate

            def segment(self, stream):
                stream.read()
                return iter(
                    [
                        types.SimpleNamespace(pcm=b"hello"),
                        types.SimpleNamespace(pcm=b"world"),
                    ]
                )

        class FakePocketsphinx:
            last_options = None
            processed = []

            def __init__(self, **options):
                FakePocketsphinx.last_options = options
                self.current_text = ""

            def start_utt(self):
                self.current_text = ""

            def process_raw(self, pcm, full_utt=False):
                FakePocketsphinx.processed.append((pcm, full_utt))
                self.current_text = pcm.decode("utf-8")

            def end_utt(self):
                pass

            def hyp(self):
                return self.current_text

            def hypothesis(self):
                return self.current_text

        fake_module = types.SimpleNamespace(
            Pocketsphinx=FakePocketsphinx,
            Segmenter=FakeSegmenter,
        )

        with patch.dict(sys.modules, {"pocketsphinx": fake_module}):
            provider = self.provider_cls.__new__(self.provider_cls)
            result = provider.transcribe(str(audio), "default", language="en")

        assert isinstance(result, TranscriptionResult)
        assert result.text == "hello world"
        assert result.language == "en"
        assert FakeSegmenter.last_sample_rate == 16000
        assert FakePocketsphinx.last_options == {}
        assert FakePocketsphinx.processed == [(b"hello", True), (b"world", True)]

    def test_validate_wav_rejects_non_mono(self, tmp_path):
        audio = tmp_path / "stereo.wav"
        with wave.open(str(audio), "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\x00\x00" * 16000)

        with wave.open(str(audio), "rb") as wav:
            with pytest.raises(ValueError, match="mono WAV"):
                self.provider_cls._validate_wav(str(audio), wav)

    def test_validate_wav_rejects_non_16_bit(self, tmp_path):
        audio = tmp_path / "eight-bit.wav"
        with wave.open(str(audio), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(1)
            wav.setframerate(16000)
            wav.writeframes(b"\x00" * 16000)

        with wave.open(str(audio), "rb") as wav:
            with pytest.raises(ValueError, match="16-bit PCM"):
                self.provider_cls._validate_wav(str(audio), wav)

    def test_missing_dependency_raises_import_error(self, tmp_path):
        audio = tmp_path / "speech.wav"
        audio.write_bytes(b"audio")

        with patch.dict(sys.modules, {"pocketsphinx": None}):
            provider = self.provider_cls.__new__(self.provider_cls)
            with pytest.raises(ImportError, match="sapat\\[pocketsphinx\\]"):
                provider.transcribe(str(audio), "default")
