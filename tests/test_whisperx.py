import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from sapat.script import main
from sapat.transcription.whisperx import WhisperXTranscription


class WhisperXTranscriptionTests(unittest.TestCase):
    def test_missing_binary_is_reported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio = Path(tmpdir) / "sample.mp3"
            audio.write_bytes(b"audio")

            with patch.dict(os.environ, {"WHISPERX_BINARY": "/missing/whisperx"}, clear=True):
                transcriber = WhisperXTranscription()

            with self.assertRaisesRegex(ValueError, "WhisperX executable"):
                transcriber.transcribe_audio(str(audio))

    def test_transcribe_audio_builds_whisperx_command_and_reads_txt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio = Path(tmpdir) / "sample.mp3"
            binary = Path(tmpdir) / "whisperx"
            audio.write_bytes(b"audio")
            binary.write_text("#!/bin/sh\n", encoding="utf-8")

            def fake_run(command, stdout, stderr, universal_newlines):
                output_dir = command[command.index("--output_dir") + 1]
                Path(output_dir, "sample.txt").write_text(
                    "Aligned transcript from WhisperX\n",
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            with patch.dict(
                os.environ,
                {
                    "WHISPERX_BINARY": str(binary),
                    "WHISPERX_MODEL": "small",
                    "WHISPERX_DEVICE": "cpu",
                    "WHISPERX_COMPUTE_TYPE": "int8",
                    "WHISPERX_BATCH_SIZE": "2",
                    "WHISPERX_HF_TOKEN": "hf_test",
                    "WHISPERX_DIARIZE": "true",
                    "WHISPERX_MIN_SPEAKERS": "1",
                    "WHISPERX_MAX_SPEAKERS": "3",
                    "WHISPERX_EXTRA_ARGS": "--threads 2",
                },
                clear=True,
            ):
                transcriber = WhisperXTranscription(temperature=0.2)

            with patch(
                "sapat.transcription.whisperx.subprocess.run",
                side_effect=fake_run,
            ) as run:
                transcript = transcriber.transcribe_audio(
                    str(audio),
                    language="en",
                    prompt="Daytona and Sapat",
                    temperature=0.1,
                )

            command = run.call_args.args[0]
            self.assertEqual(transcript, "Aligned transcript from WhisperX")
            self.assertEqual(command[0], str(binary))
            self.assertIn(str(audio), command)
            self.assertIn("--output_format", command)
            self.assertIn("txt", command)
            self.assertIn("--language", command)
            self.assertIn("en", command)
            self.assertIn("--initial_prompt", command)
            self.assertIn("Daytona and Sapat", command)
            self.assertIn("--hf_token", command)
            self.assertIn("hf_test", command)
            self.assertIn("--diarize", command)
            self.assertIn("--min_speakers", command)
            self.assertIn("1", command)
            self.assertIn("--max_speakers", command)
            self.assertIn("3", command)
            self.assertIn("--threads", command)
            self.assertIn("2", command)

    def test_process_file_converts_to_mp3_and_writes_transcript(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video = Path(tmpdir) / "meeting.mp4"
            video.write_bytes(b"video")

            transcriber = WhisperXTranscription()

            def fake_convert(input_file, output_file, quality):
                Path(output_file).write_bytes(b"mp3")

            with patch.object(transcriber, "convert_to_mp3", side_effect=fake_convert):
                with patch.object(
                    transcriber,
                    "transcribe_audio",
                    return_value="Transcript text",
                ):
                    transcriber.process_file(video, "en", None, 0.0, "M", False)

            self.assertEqual(
                video.with_suffix(".txt").read_text(encoding="utf-8"),
                "Transcript text",
            )
            self.assertFalse(video.with_suffix(".mp3").exists())

    def test_cli_accepts_whisperx_api(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video = Path(tmpdir) / "meeting.mp4"
            video.write_bytes(b"video")

            with patch(
                "sapat.script.WhisperXTranscription.process_file"
            ) as process_file:
                result = CliRunner().invoke(
                    main,
                    [str(video), "--api", "whisperx"],
                )

            self.assertEqual(result.exit_code, 0)
            process_file.assert_called_once()


if __name__ == "__main__":
    unittest.main()
