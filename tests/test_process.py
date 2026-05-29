# ABOUTME: Tests for the shared file processing pipeline
# ABOUTME: Covers temporary conversion cleanup and original-file preservation

from pathlib import Path
from unittest.mock import patch

from sapat.process import process_file
from sapat.providers.base import (
    AudioFormat,
    ProviderConfig,
    TranscriptionProvider,
    TranscriptionResult,
)


class FakeWavProvider(TranscriptionProvider):
    name = "fake_wav"
    config = ProviderConfig(
        preferred_format=AudioFormat.WAV,
        max_file_size_mb=float("inf"),
    )

    def transcribe(
        self, audio_file, model, language="en", prompt=None, temperature=0, **kwargs
    ):
        self.last_audio_file = audio_file
        return TranscriptionResult(text="safe transcript")


def test_process_file_preserves_original_when_already_preferred_format(tmp_path):
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"original wav")
    provider = FakeWavProvider()

    with patch("sapat.process.should_split_file", return_value=False), patch(
        "sapat.process.convert_audio"
    ) as mock_convert:
        process_file(str(audio), provider, "default", "en", None, 0, "L", False)

    assert audio.exists()
    assert audio.read_bytes() == b"original wav"
    assert Path(provider.last_audio_file) == audio
    assert audio.with_suffix(".txt").read_text(encoding="utf-8") == "safe transcript"
    mock_convert.assert_not_called()


def test_process_file_removes_only_created_converted_file(tmp_path):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    converted = tmp_path / "input.wav"
    provider = FakeWavProvider()

    def fake_convert(input_file, output_file, quality, target_format):
        Path(output_file).write_bytes(b"converted wav")

    with patch("sapat.process.should_split_file", return_value=False), patch(
        "sapat.process.convert_audio", side_effect=fake_convert
    ):
        process_file(str(source), provider, "default", "en", None, 0, "L", False)

    assert source.exists()
    assert not converted.exists()
    assert Path(provider.last_audio_file) == converted
    assert source.with_suffix(".txt").read_text(encoding="utf-8") == "safe transcript"
