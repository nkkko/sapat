from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from sapat.transcription.soniox import SonioxTranscription


class FakeSttClient:
    def __init__(self):
        self.transcribe_calls = []
        self.wait_calls = []
        self.destroy_calls = []

    def transcribe(self, **kwargs):
        self.transcribe_calls.append(kwargs)
        return SimpleNamespace(id="transcription-1")

    def wait(self, transcription_id):
        self.wait_calls.append(transcription_id)

    def get_transcript(self, transcription_id):
        return SimpleNamespace(text=f"transcript for {transcription_id}")

    def destroy(self, transcription_id):
        self.destroy_calls.append(transcription_id)


class FakeSonioxClient:
    def __init__(self):
        self.stt = FakeSttClient()


class SonioxTranscriptionTest(TestCase):
    def test_transcribes_local_file_and_cleans_remote_job(self):
        audio_file = Path("sample.mp3")
        audio_file.write_bytes(b"fake audio")
        try:
            with patch("sapat.transcription.soniox.SonioxClient", FakeSonioxClient):
                transcriber = SonioxTranscription(temperature=0.3)
                result = transcriber.transcribe_audio(str(audio_file))

            self.assertEqual(result, {"text": "transcript for transcription-1"})
            self.assertEqual(
                transcriber.client.stt.transcribe_calls,
                [{"model": "stt-async-v4", "file": "sample.mp3"}],
            )
            self.assertEqual(transcriber.client.stt.wait_calls, ["transcription-1"])
            self.assertEqual(transcriber.client.stt.destroy_calls, ["transcription-1"])
        finally:
            audio_file.unlink(missing_ok=True)

    def test_rejects_unsupported_audio_extension(self):
        with patch("sapat.transcription.soniox.SonioxClient", FakeSonioxClient):
            transcriber = SonioxTranscription(temperature=0.3)

        with self.assertRaises(ValueError):
            transcriber._validate_audio_file("sample.txt")
