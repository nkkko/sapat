import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from click.testing import CliRunner

from sapat.script import main
from sapat.transcription.moonshine import MoonshineTranscription


class FakeLine:
    def __init__(self, text, start_time):
        self.text = text
        self.start_time = start_time


class FakeTranscriber:
    last_instance = None

    def __init__(self, model_path, model_arch, update_interval):
        self.model_path = model_path
        self.model_arch = model_arch
        self.update_interval = update_interval
        self.closed = False
        FakeTranscriber.last_instance = self

    def transcribe_without_streaming(self, audio_data, sample_rate):
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        return types.SimpleNamespace(
            lines=[
                FakeLine("second line", 2.0),
                FakeLine("first line", 1.0),
            ]
        )

    def close(self):
        self.closed = True


class MoonshineTranscriptionTests(unittest.TestCase):
    def test_transcript_lines_are_written_in_chronological_order(self):
        transcript = types.SimpleNamespace(
            lines=[
                FakeLine("later", 4.0),
                FakeLine("earlier", 1.0),
                FakeLine(" ", 2.0),
            ]
        )

        result = MoonshineTranscription._transcript_to_text(transcript)

        self.assertEqual(result, "earlier\nlater")

    def test_transcribe_audio_uses_moonshine_voice_package(self):
        fake_module = types.SimpleNamespace(
            Transcriber=FakeTranscriber,
            get_model_for_language=lambda language: (f"/models/{language}", "base"),
            load_wav_file=lambda audio_file: ([0.1, -0.1], 16000),
        )

        with mock.patch.dict(sys.modules, {"moonshine_voice": fake_module}):
            result = MoonshineTranscription(update_interval=0.25).transcribe_audio(
                "speech.wav",
                language="en",
            )

        self.assertEqual(result, "first line\nsecond line")
        self.assertEqual(FakeTranscriber.last_instance.model_path, "/models/en")
        self.assertEqual(FakeTranscriber.last_instance.model_arch, "base")
        self.assertEqual(FakeTranscriber.last_instance.update_interval, 0.25)
        self.assertTrue(FakeTranscriber.last_instance.closed)

    def test_missing_moonshine_dependency_has_actionable_message(self):
        with mock.patch.dict(sys.modules, {"moonshine_voice": None}):
            with self.assertRaises(ImportError) as raised:
                MoonshineTranscription().transcribe_audio("speech.wav", language="en")

        self.assertIn("pip install 'sapat[moonshine]'", str(raised.exception))

    def test_process_file_converts_to_temporary_wav_and_removes_it(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "demo.mp4"
            video_path.write_bytes(b"not really video")
            wav_path = video_path.with_suffix(".wav")
            txt_path = video_path.with_suffix(".txt")
            transcriber = MoonshineTranscription()

            def fake_convert(input_file, output_file):
                self.assertEqual(input_file, str(video_path))
                Path(output_file).write_bytes(b"wav")

            with mock.patch.object(transcriber, "convert_to_wav", side_effect=fake_convert):
                with mock.patch.object(transcriber, "transcribe_audio", return_value="hello"):
                    transcriber.process_file(video_path, "en", None, 0.3, "M", False)

            self.assertEqual(txt_path.read_text(encoding="utf-8"), "hello")
            self.assertFalse(wav_path.exists())

    def test_cli_exposes_moonshine_api_choice(self):
        result = CliRunner().invoke(main, ["--help"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("moonshine", result.output)


if __name__ == "__main__":
    unittest.main()
