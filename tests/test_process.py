# ABOUTME: Tests for the file processing orchestration layer
# ABOUTME: Verifies conversion cleanup does not remove user-owned source audio

from contextlib import contextmanager

import pytest

from sapat.process import process_file
from sapat.providers.base import (
    AudioFormat,
    ProviderConfig,
    TranscriptionProvider,
    TranscriptionResult,
)


class DummyProvider(TranscriptionProvider):
    name = "dummy"
    config = ProviderConfig(preferred_format=AudioFormat.MP3)

    def transcribe(
        self, audio_file, model, language="en", prompt=None, temperature=0, **kwargs
    ):
        return TranscriptionResult(text=f"transcribed {audio_file}")


class DummySpinner:
    def succeed(self, message):
        pass

    def fail(self, message):
        pass


@contextmanager
def dummy_spinner_context(message):
    yield DummySpinner()


@pytest.fixture(autouse=True)
def no_spinner(monkeypatch):
    monkeypatch.setattr("sapat.process.spinner_context", dummy_spinner_context)


def test_process_file_keeps_existing_preferred_audio(tmp_path, monkeypatch):
    audio_file = tmp_path / "sample.mp3"
    audio_file.write_bytes(b"already mp3")

    def fail_convert(*args, **kwargs):
        raise AssertionError("conversion should not run for existing mp3")

    monkeypatch.setattr("sapat.process.convert_audio", fail_convert)

    process_file(
        str(audio_file),
        DummyProvider(),
        model="model",
        language="en",
        prompt=None,
        temperature=0,
        quality="M",
        correct=False,
    )

    assert audio_file.exists()
    assert (tmp_path / "sample.txt").read_text(encoding="utf-8") == (
        f"transcribed {audio_file}"
    )


def test_process_file_removes_converted_temp_audio(tmp_path, monkeypatch):
    video_file = tmp_path / "sample.wav"
    video_file.write_bytes(b"source audio")
    converted_file = tmp_path / "sample.mp3"

    def fake_convert(input_file, output_file, quality, fmt):
        assert input_file == str(video_file)
        assert output_file == str(converted_file)
        converted_file.write_bytes(b"converted audio")

    monkeypatch.setattr("sapat.process.convert_audio", fake_convert)

    process_file(
        str(video_file),
        DummyProvider(),
        model="model",
        language="en",
        prompt=None,
        temperature=0,
        quality="M",
        correct=False,
    )

    assert video_file.exists()
    assert not converted_file.exists()
    assert (tmp_path / "sample.txt").read_text(encoding="utf-8") == (
        f"transcribed {converted_file}"
    )
