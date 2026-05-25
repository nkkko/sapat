import json
import os
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from sapat.transcription.oracle import OracleSpeechTranscription


class OracleSpeechTranscriptionTests(unittest.TestCase):
    def test_transcribe_audio_uploads_job_polls_and_reads_output(self):
        audio_file = Path("sample.mp3")
        audio_file.write_bytes(b"fake mp3")
        object_client = Mock()
        speech_client = Mock()
        speech_client.create_transcription_job.return_value.data = types.SimpleNamespace(id="job1")
        speech_client.list_transcription_tasks.return_value.data = types.SimpleNamespace(
            items=[
                types.SimpleNamespace(
                    id="task1",
                    lifecycle_state="SUCCEEDED",
                )
            ]
        )
        speech_client.get_transcription_task.return_value.data = types.SimpleNamespace(
            output_location=types.SimpleNamespace(
                namespace_name="ns",
                bucket_name="out",
                object_names=["sapat-transcripts/job1/sample.json"],
            ),
        )
        object_client.get_object.return_value.data.content = json.dumps(
            {"transcriptions": [{"transcription": "oracle transcript"}]}
        ).encode("utf-8")
        oci = self._fake_oci(object_client, speech_client)

        env = {
            "OCI_COMPARTMENT_ID": "ocid1.compartment.oc1..example",
            "OCI_OBJECT_STORAGE_NAMESPACE": "ns",
            "OCI_SPEECH_INPUT_BUCKET": "in",
            "OCI_SPEECH_OUTPUT_BUCKET": "out",
            "OCI_SPEECH_POLL_INTERVAL_SECONDS": "0",
        }

        try:
            with patch.dict(os.environ, env, clear=False), patch.object(
                OracleSpeechTranscription, "_load_oci", return_value=oci
            ):
                transcript = OracleSpeechTranscription(temperature=0.3).transcribe_audio(
                    str(audio_file),
                    language="en",
                )
        finally:
            if audio_file.exists():
                audio_file.unlink()

        self.assertEqual(transcript, "oracle transcript")
        object_client.put_object.assert_called_once()
        speech_client.create_transcription_job.assert_called_once()
        speech_client.list_transcription_tasks.assert_called_once_with("job1")
        speech_client.get_transcription_task.assert_called_once_with("job1", "task1")
        object_client.get_object.assert_called_once_with("ns", "out", "sapat-transcripts/job1/sample.json")

    def test_transcribe_audio_requires_oracle_settings(self):
        audio_file = Path("sample.mp3")
        audio_file.write_bytes(b"fake mp3")
        try:
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(ValueError, "OCI_COMPARTMENT_ID"):
                    OracleSpeechTranscription(temperature=0.3).transcribe_audio(str(audio_file))
        finally:
            if audio_file.exists():
                audio_file.unlink()

    def test_oracle_model_requires_locale_language(self):
        with patch.dict(
            os.environ,
            {
                "OCI_COMPARTMENT_ID": "ocid1.compartment.oc1..example",
                "OCI_OBJECT_STORAGE_NAMESPACE": "ns",
                "OCI_SPEECH_INPUT_BUCKET": "in",
                "OCI_SPEECH_OUTPUT_BUCKET": "out",
            },
            clear=True,
        ):
            transcriber = OracleSpeechTranscription(temperature=0.3)

        self.assertEqual(transcriber._normalize_language("es"), "es-ES")
        with self.assertRaisesRegex(ValueError, "requires an explicit locale"):
            transcriber._normalize_language("auto")

    def _fake_oci(self, object_client, speech_client):
        models = types.SimpleNamespace(
            CreateTranscriptionJobDetails=self._record_factory("CreateTranscriptionJobDetails"),
            ObjectListInlineInputLocation=self._record_factory("ObjectListInlineInputLocation"),
            ObjectLocation=self._record_factory("ObjectLocation"),
            OutputLocation=self._record_factory("OutputLocation"),
            TranscriptionModelDetails=self._record_factory("TranscriptionModelDetails"),
        )
        return types.SimpleNamespace(
            config=types.SimpleNamespace(from_file=Mock(return_value={"region": "us-ashburn-1"})),
            object_storage=types.SimpleNamespace(ObjectStorageClient=Mock(return_value=object_client)),
            ai_speech=types.SimpleNamespace(
                AIServiceSpeechClient=Mock(return_value=speech_client),
                models=models,
            ),
        )

    @staticmethod
    def _record_factory(name):
        def factory(**kwargs):
            return types.SimpleNamespace(_model_name=name, **kwargs)

        return factory


if __name__ == "__main__":
    unittest.main()
