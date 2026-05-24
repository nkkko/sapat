import io
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from click.testing import CliRunner

from sapat.transcription.aws import AWSTranscribeTranscription


class AWSTranscribeTranscriptionTests(unittest.TestCase):
    def _audio_file(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp.write(b"audio")
        tmp.close()
        self.addCleanup(lambda: Path(tmp.name).exists() and Path(tmp.name).unlink())
        return tmp.name

    def _fake_boto3(self, s3, transcribe):
        return types.SimpleNamespace(client=Mock(side_effect=[s3, transcribe]))

    @patch.dict(os.environ, {"AWS_TRANSCRIBE_REGION": "us-west-2"}, clear=True)
    def test_missing_input_bucket_is_rejected(self):
        transcriber = AWSTranscribeTranscription(temperature=0.0)

        with self.assertRaisesRegex(ValueError, "AWS_TRANSCRIBE_S3_BUCKET"):
            transcriber.transcribe_audio(self._audio_file())

    @patch("sapat.transcription.aws.time.sleep")
    @patch("sapat.transcription.aws.requests.get")
    @patch.dict(
        os.environ,
        {
            "AWS_TRANSCRIBE_S3_BUCKET": "input-bucket",
            "AWS_TRANSCRIBE_REGION": "us-west-2",
            "AWS_TRANSCRIBE_POLL_INTERVAL": "0",
        },
        clear=True,
    )
    def test_transcribes_with_service_managed_transcript_uri(self, mock_get, _mock_sleep):
        s3 = Mock()
        transcribe = Mock()
        transcribe.get_transcription_job.side_effect = [
            {"TranscriptionJob": {"TranscriptionJobStatus": "IN_PROGRESS"}},
            {
                "TranscriptionJob": {
                    "TranscriptionJobStatus": "COMPLETED",
                    "Transcript": {"TranscriptFileUri": "https://example.com/result.json"},
                }
            },
        ]
        mock_response = Mock()
        mock_response.json.return_value = {
            "results": {"transcripts": [{"transcript": "hello from aws"}]}
        }
        mock_get.return_value = mock_response

        with patch("sapat.transcription.aws.boto3", self._fake_boto3(s3, transcribe)):
            result = AWSTranscribeTranscription(temperature=0.0).transcribe_audio(
                self._audio_file(), language="en"
            )

        self.assertEqual(result["text"], "hello from aws")
        s3.upload_file.assert_called_once()
        s3.delete_object.assert_called_once()
        start_args = transcribe.start_transcription_job.call_args.kwargs
        self.assertEqual(start_args["LanguageCode"], "en-US")
        self.assertEqual(start_args["MediaFormat"], "mp3")
        self.assertTrue(start_args["Media"]["MediaFileUri"].startswith("s3://input-bucket/sapat/"))
        mock_response.raise_for_status.assert_called_once()

    @patch.dict(
        os.environ,
        {
            "AWS_TRANSCRIBE_S3_BUCKET": "input-bucket",
            "AWS_TRANSCRIBE_OUTPUT_BUCKET": "output-bucket",
            "AWS_TRANSCRIBE_OUTPUT_PREFIX": "transcripts/",
            "AWS_TRANSCRIBE_DELETE_MEDIA": "false",
        },
        clear=True,
    )
    def test_reads_transcript_from_configured_output_bucket(self):
        s3 = Mock()
        transcribe = Mock()
        transcribe.get_transcription_job.return_value = {
            "TranscriptionJob": {
                "TranscriptionJobStatus": "COMPLETED",
                "Transcript": {"TranscriptFileUri": "unused"},
            }
        }
        payload = {"results": {"transcripts": [{"transcript": "from output bucket"}]}}
        s3.get_object.return_value = {"Body": io.BytesIO(json.dumps(payload).encode("utf-8"))}

        with patch("sapat.transcription.aws.boto3", self._fake_boto3(s3, transcribe)):
            result = AWSTranscribeTranscription(temperature=0.0).transcribe_audio(self._audio_file())

        self.assertEqual(result["text"], "from output bucket")
        start_args = transcribe.start_transcription_job.call_args.kwargs
        self.assertEqual(start_args["OutputBucketName"], "output-bucket")
        self.assertTrue(start_args["OutputKey"].startswith("transcripts/"))
        s3.delete_object.assert_not_called()

    @patch.dict(os.environ, {"AWS_TRANSCRIBE_S3_BUCKET": "input-bucket"}, clear=True)
    def test_failed_job_raises_failure_reason(self):
        s3 = Mock()
        transcribe = Mock()
        transcribe.get_transcription_job.return_value = {
            "TranscriptionJob": {
                "TranscriptionJobStatus": "FAILED",
                "FailureReason": "unsupported media",
            }
        }

        with patch("sapat.transcription.aws.boto3", self._fake_boto3(s3, transcribe)):
            with self.assertRaisesRegex(RuntimeError, "unsupported media"):
                AWSTranscribeTranscription(temperature=0.0).transcribe_audio(self._audio_file())

    def test_cli_routes_aws_provider(self):
        from sapat import script

        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".mp4") as video:
            with patch.object(script, "AWSTranscribeTranscription") as provider:
                result = runner.invoke(script.main, [video.name, "--api", "aws"])

        self.assertEqual(result.exit_code, 0)
        provider.assert_called_once_with(temperature=0.3)
        provider.return_value.process_file.assert_called_once()


if __name__ == "__main__":
    unittest.main()
