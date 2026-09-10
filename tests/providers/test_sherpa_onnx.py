# ABOUTME: Tests for the local sherpa-onnx transcription provider
# ABOUTME: Exercises model-family setup and decoding without model downloads

import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from sapat.cli import main
from sapat.providers import _registry, register
from sapat.providers.base import TranscriptionResult
from sapat.providers.sherpa_onnx import SherpaOnnxProvider


def _wav(path: Path) -> Path:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x00\x00" * 160)
    return path


def _files(tmp_path: Path, *names: str):
    values = {}
    for name in names:
        path = tmp_path / name
        path.write_bytes(b"model")
        values[name] = str(path)
    return values


class _FakeStream:
    def __init__(self):
        self.result = types.SimpleNamespace(text="  local transcript  ")

    def accept_waveform(self, sample_rate, samples):
        self.sample_rate = sample_rate
        self.samples = samples


class _FakeRecognizer:
    def __init__(self, family, kwargs):
        self.family = family
        self.kwargs = kwargs

    def create_stream(self):
        self.stream = _FakeStream()
        return self.stream

    def decode_stream(self, stream):
        self.decoded = stream


class _Factory:
    calls = []

    @classmethod
    def _create(cls, family, **kwargs):
        recognizer = _FakeRecognizer(family, kwargs)
        cls.calls.append(recognizer)
        return recognizer

    from_sense_voice = classmethod(
        lambda cls, **kwargs: cls._create("sense_voice", **kwargs)
    )
    from_whisper = classmethod(lambda cls, **kwargs: cls._create("whisper", **kwargs))
    from_transducer = classmethod(
        lambda cls, **kwargs: cls._create("transducer", **kwargs)
    )


@pytest.fixture(autouse=True)
def fake_sherpa(monkeypatch):
    _Factory.calls.clear()
    monkeypatch.setitem(
        sys.modules,
        "sherpa_onnx",
        types.SimpleNamespace(OfflineRecognizer=_Factory),
    )
    _registry.clear()
    yield
    _registry.clear()


def test_sense_voice_transcription_is_local_and_typed(tmp_path):
    audio = _wav(tmp_path / "speech.wav")
    files = _files(tmp_path, "sense.onnx", "tokens.txt")

    result = SherpaOnnxProvider().transcribe(
        str(audio),
        "sense_voice",
        model_path=files["sense.onnx"],
        tokens=files["tokens.txt"],
        num_threads=3,
    )

    assert isinstance(result, TranscriptionResult)
    assert result.text == "local transcript"
    assert result.language == "en"
    assert result.duration == pytest.approx(0.01)
    call = _Factory.calls[-1]
    assert call.family == "sense_voice"
    assert call.kwargs["num_threads"] == 3
    assert call.kwargs["language"] == "en"
    assert call.kwargs["provider"] == "cpu"
    assert call.stream.sample_rate == 16000
    assert call.stream.samples.dtype == np.float32


def test_whisper_uses_explicit_bundle_and_language(tmp_path):
    audio = _wav(tmp_path / "speech.wav")
    files = _files(tmp_path, "encoder.onnx", "decoder.onnx", "tokens.txt")

    SherpaOnnxProvider().transcribe(
        str(audio),
        "whisper",
        language="de",
        encoder=files["encoder.onnx"],
        decoder=files["decoder.onnx"],
        tokens=files["tokens.txt"],
    )

    call = _Factory.calls[-1]
    assert call.family == "whisper"
    assert call.kwargs["language"] == "de"
    assert call.kwargs["task"] == "transcribe"


def test_transducer_requires_joiner(tmp_path):
    audio = _wav(tmp_path / "speech.wav")
    files = _files(tmp_path, "encoder.onnx", "decoder.onnx", "tokens.txt")

    with pytest.raises(ValueError, match="joiner"):
        SherpaOnnxProvider().transcribe(
            str(audio),
            "transducer",
            encoder=files["encoder.onnx"],
            decoder=files["decoder.onnx"],
            tokens=files["tokens.txt"],
        )


def test_rejects_non_pcm_or_stereo_wav(tmp_path):
    audio = tmp_path / "stereo.wav"
    with wave.open(str(audio), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\x00\x00\x00\x00" * 4)
    files = _files(tmp_path, "sense.onnx", "tokens.txt")

    with pytest.raises(ValueError, match="mono 16-bit PCM"):
        SherpaOnnxProvider().transcribe(
            str(audio),
            "sense_voice",
            model_path=files["sense.onnx"],
            tokens=files["tokens.txt"],
        )


def test_rejects_unknown_family(tmp_path):
    audio = _wav(tmp_path / "speech.wav")
    with pytest.raises(ValueError, match="Unsupported sherpa-onnx model family"):
        SherpaOnnxProvider().transcribe(str(audio), "mystery")


def test_registry_and_cli_route_to_provider(tmp_path, monkeypatch):
    audio = _wav(tmp_path / "speech.wav")
    routed = []
    register(SherpaOnnxProvider)
    monkeypatch.setattr(
        "sapat.cli.process_file",
        lambda input_file, provider, model, *args: routed.append(
            (input_file, provider.name, model)
        ),
    )

    result = CliRunner().invoke(
        main,
        [str(audio), "--provider", "sherpa_onnx", "--model", "sense_voice"],
    )

    assert result.exit_code == 0, result.output
    assert routed == [(str(audio), "sherpa_onnx", "sense_voice")]
