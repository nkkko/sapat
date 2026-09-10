# ABOUTME: Regression tests for source preservation and temporary audio cleanup

import shutil
import subprocess
import wave
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock

import pytest

from sapat.process import process_file
from sapat.providers.base import AudioFormat, ProviderConfig, TranscriptionResult


@pytest.fixture(autouse=True)
def quiet_spinner(monkeypatch):
    monkeypatch.setattr(
        "sapat.process.spinner_context", lambda text: nullcontext(Mock())
    )


def write_wav(path, channels=1):
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(44100)
        audio.writeframes(b"\x01\x00" * channels * 4410)


def transcribe(source, provider, correct=False):
    process_file(str(source), provider, "test", "en", None, 0, "M", correct)


def make_provider():
    provider = Mock()
    provider.name = "test"
    provider.config = ProviderConfig(
        preferred_format=AudioFormat.WAV,
        max_file_size_mb=float("inf"),
        supports_correction=True,
    )
    provider.transcribe.return_value = TranscriptionResult(text="local transcript")
    return provider


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="requires ffmpeg")
@pytest.mark.parametrize("channels", [1, 2])
def test_wav_source_is_preserved_and_normalized(tmp_path, channels):
    source = tmp_path / "speech.wav"
    write_wav(source, channels)
    original = source.read_bytes()
    provider = make_provider()

    def inspect_audio(path, **kwargs):
        with wave.open(path, "rb") as audio:
            assert (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) == (
                1, 2, 16000
            )
            assert audio.getnframes() > 0
        return TranscriptionResult(text="local transcript")

    provider.transcribe.side_effect = inspect_audio
    transcribe(source, provider)

    assert source.read_bytes() == original
    assert source.with_suffix(".txt").read_text() == "local transcript"
    temporary = Path(provider.transcribe.call_args.args[0])
    assert not temporary.exists()
    assert not temporary.parent.exists()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="requires ffmpeg")
def test_existing_sibling_wav_is_neither_used_nor_deleted(tmp_path):
    sibling = tmp_path / "speech.wav"
    write_wav(sibling)
    source = tmp_path / "speech.flac"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(sibling), str(source)],
        check=True,
    )
    original = source.read_bytes()
    sibling.write_bytes(b"unrelated existing WAV must remain untouched")
    sibling_original = sibling.read_bytes()
    provider = make_provider()

    transcribe(source, provider)

    assert source.read_bytes() == original
    assert sibling.read_bytes() == sibling_original
    assert source.with_suffix(".txt").read_text() == "local transcript"
    temporary = Path(provider.transcribe.call_args.args[0])
    assert temporary != sibling
    assert not temporary.parent.exists()


@pytest.mark.parametrize("failure", ["conversion", "transcription", "correction", "output"])
def test_failed_processing_preserves_source_and_cleans_temporary_audio(
    tmp_path, monkeypatch, failure
):
    source = tmp_path / "speech.wav"
    write_wav(source)
    original = source.read_bytes()
    converted = []
    provider = make_provider()

    def convert(input_file, output_file, *args):
        path = Path(output_file)
        converted.append(path)
        path.write_bytes(original)
        if failure == "conversion":
            raise RuntimeError("conversion failed")

    monkeypatch.setattr("sapat.process.convert_audio", convert)
    if failure == "transcription":
        provider.transcribe.side_effect = RuntimeError("transcription failed")
    elif failure == "correction":
        provider.correct_transcript.side_effect = RuntimeError("correction failed")
    elif failure == "output":
        source.with_suffix(".txt").mkdir()

    if failure == "conversion":
        transcribe(source, provider)
        provider.transcribe.assert_not_called()
    else:
        expected_error = OSError if failure == "output" else RuntimeError
        with pytest.raises(expected_error):
            transcribe(source, provider, correct=failure == "correction")

    assert source.read_bytes() == original
    assert len(converted) == 1
    assert not converted[0].exists()
    assert not converted[0].parent.exists()
