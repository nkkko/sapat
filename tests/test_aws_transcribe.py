import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

from sapat.transcription.aws_transcribe import AWSTranscribeTranscription


class AWSTranscribeTranscriptionTest(unittest.TestCase):
    def make_audio_file(self, suffix=".mp3"):
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(b"audio-bytes")
        tmp.close()
        self.addCleanup(lambda: Path(tmp.name).exists() and Path(tmp.name).unlink())
        return tmp.name

    @patch.dict(os.environ, {
        "AWS_TRANSCRIBE_S3_BUCKET": "media-bucket",
        "AWS_TRANSCRIBE_REGION": "us-west-2",
        "AWS_TRANSCRIBE_POLL_SECONDS": "0",
    }, clear=True)
    @patch("sapat.transcription.aws_transcribe.requests.get")
    @patch("sapat.transcription.aws_transcribe.boto3.client")
    @patch("sapat.transcription.aws_transcribe.time.sleep")
    def test_uploads_starts_job_and_returns_transcript(self, sleep, boto_client, get):
        audio_file = self.make_audio_file()
        s3 = Mock()
        transcribe = Mock()
        boto_client.side_effect = [s3, transcribe]
        transcribe.get_transcription_job.side_effect = [
            {"TranscriptionJob": {"TranscriptionJobStatus": "IN_PROGRESS"}},
            {
                "TranscriptionJob": {
                    "TranscriptionJobStatus": "COMPLETED",
                    "Transcript": {"TranscriptFileUri": "https://example.com/transcript.json"},
                }
            },
        ]
        get.return_value = Mock(
            status_code=200,
            json=lambda: {"results": {"transcripts": [{"transcript": "aws transcript"}]}},
        )

        result = AWSTranscribeTranscription(temperature=0.3).transcribe_audio(
            audio_file,
            language="en",
        )

        self.assertEqual(result, "aws transcript")
        boto_client.assert_has_calls([
            call("s3", region_name="us-west-2"),
            call("transcribe", region_name="us-west-2"),
        ])
        s3.upload_file.assert_called_once()
        self.assertEqual(s3.upload_file.call_args.args[1], "media-bucket")
        start_args = transcribe.start_transcription_job.call_args.kwargs
        self.assertEqual(start_args["LanguageCode"], "en-US")
        self.assertEqual(start_args["MediaFormat"], "mp3")
        self.assertTrue(
            start_args["Media"]["MediaFileUri"].startswith(
                "s3://media-bucket/sapat-transcribe/"
            )
        )
        sleep.assert_called_once_with(0.0)

    @patch.dict(os.environ, {}, clear=True)
    @patch("sapat.transcription.aws_transcribe.boto3.client")
    def test_requires_s3_bucket(self, boto_client):
        boto_client.side_effect = [Mock(), Mock()]
        audio_file = self.make_audio_file()

        with self.assertRaisesRegex(ValueError, "AWS_TRANSCRIBE_S3_BUCKET"):
            AWSTranscribeTranscription(temperature=0.3).transcribe_audio(audio_file)

    @patch.dict(os.environ, {
        "AWS_TRANSCRIBE_S3_BUCKET": "media-bucket",
        "AWS_TRANSCRIBE_OUTPUT_BUCKET": "output-bucket",
        "AWS_TRANSCRIBE_OUTPUT_PREFIX": "jobs",
        "AWS_TRANSCRIBE_IDENTIFY_LANGUAGE": "true",
    }, clear=True)
    @patch("sapat.transcription.aws_transcribe.boto3.client")
    def test_start_job_can_use_language_identification_and_output_bucket(self, boto_client):
        boto_client.side_effect = [Mock(), Mock()]
        provider = AWSTranscribeTranscription(temperature=0.3)

        provider._start_job("job-name", "media/key.mp3", "media/key.mp3", language=None)

        start_args = provider.transcribe.start_transcription_job.call_args.kwargs
        self.assertTrue(start_args["IdentifyLanguage"])
        self.assertNotIn("LanguageCode", start_args)
        self.assertEqual(start_args["OutputBucketName"], "output-bucket")
        self.assertEqual(start_args["OutputKey"], "jobs/job-name.json")


    @patch.dict(os.environ, {"AWS_TRANSCRIBE_S3_BUCKET": "media-bucket"}, clear=True)
    @patch("sapat.transcription.aws_transcribe.boto3.client")
    def test_reads_transcript_from_s3_uri(self, boto_client):
        s3 = Mock()
        transcribe = Mock()
        boto_client.side_effect = [s3, transcribe]
        s3.get_object.return_value = {
            "Body": io.BytesIO(b'{"results":{"transcripts":[{"transcript":"from s3"}]}}')
        }
        provider = AWSTranscribeTranscription(temperature=0.3)

        payload = provider._load_transcript_payload("s3://output-bucket/jobs/job.json")

        self.assertEqual(provider._extract_transcript(payload), "from s3")
        s3.get_object.assert_called_once_with(Bucket="output-bucket", Key="jobs/job.json")

    @patch.dict(os.environ, {"AWS_TRANSCRIBE_S3_BUCKET": "media-bucket"}, clear=True)
    @patch("sapat.transcription.aws_transcribe.boto3.client")
    def test_failed_job_raises_reason(self, boto_client):
        boto_client.side_effect = [Mock(), Mock()]
        provider = AWSTranscribeTranscription(temperature=0.3)
        provider.transcribe.get_transcription_job.return_value = {
            "TranscriptionJob": {
                "TranscriptionJobStatus": "FAILED",
                "FailureReason": "unsupported format",
            }
        }

        with self.assertRaisesRegex(Exception, "unsupported format"):
            provider._wait_for_transcript("job-name")


if __name__ == "__main__":
    unittest.main()
