import json
import os
import re
import time
import uuid
from pathlib import Path

import requests
from dotenv import load_dotenv

from .base import TranscriptionBase

try:
    import boto3
except ImportError:  # pragma: no cover - exercised when users install without boto3
    boto3 = None


load_dotenv(".env")


class AWSTranscribeTranscription(TranscriptionBase):
    """
    Amazon Transcribe implementation using S3-backed batch jobs.
    """

    def __init__(self, temperature: float):
        self.region = os.getenv("AWS_TRANSCRIBE_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
        self.input_bucket = os.getenv("AWS_TRANSCRIBE_S3_BUCKET")
        self.input_prefix = os.getenv("AWS_TRANSCRIBE_S3_PREFIX", "sapat/")
        self.output_bucket = os.getenv("AWS_TRANSCRIBE_OUTPUT_BUCKET")
        self.output_prefix = os.getenv("AWS_TRANSCRIBE_OUTPUT_PREFIX", "sapat-transcripts/")
        self.default_language_code = os.getenv("AWS_TRANSCRIBE_LANGUAGE_CODE", "en-US")
        self.poll_interval = float(os.getenv("AWS_TRANSCRIBE_POLL_INTERVAL", "5"))
        self.timeout_seconds = int(os.getenv("AWS_TRANSCRIBE_TIMEOUT_SECONDS", "900"))
        self.delete_uploaded_media = os.getenv("AWS_TRANSCRIBE_DELETE_MEDIA", "true").lower() not in {
            "0",
            "false",
            "no",
        }
        self.temperature = temperature

    def transcribe_audio(self, audio_file: str, **kwargs):
        """
        Uploads audio to S3, starts an Amazon Transcribe job, and returns the transcript text.
        """
        self._validate_audio_file(audio_file)

        if not self.input_bucket:
            raise ValueError("AWS_TRANSCRIBE_S3_BUCKET must be set for AWS transcription.")

        s3 = self._client("s3")
        transcribe = self._client("transcribe")

        audio_path = Path(audio_file)
        media_format = self._media_format(audio_path)
        job_name = self._job_name(audio_path)
        input_key = self._join_s3_key(self.input_prefix, f"{job_name}.{media_format}")

        output_key = None
        start_args = {
            "TranscriptionJobName": job_name,
            "LanguageCode": self._language_code(kwargs.get("language")),
            "MediaFormat": media_format,
            "Media": {"MediaFileUri": f"s3://{self.input_bucket}/{input_key}"},
        }

        if self.output_bucket:
            output_key = self._join_s3_key(self.output_prefix, f"{job_name}.json")
            start_args["OutputBucketName"] = self.output_bucket
            start_args["OutputKey"] = output_key

        try:
            s3.upload_file(str(audio_path), self.input_bucket, input_key)
            transcribe.start_transcription_job(**start_args)
            job = self._wait_for_job(transcribe, job_name)
            transcript_json = self._load_transcript_json(s3, job, output_key)
            return {"text": self._extract_transcript_text(transcript_json), "raw": transcript_json}
        finally:
            if self.delete_uploaded_media:
                try:
                    s3.delete_object(Bucket=self.input_bucket, Key=input_key)
                except Exception:
                    pass

    def _client(self, service_name: str):
        if boto3 is None:
            raise ImportError("boto3 is required for AWS transcription. Install Sapat with boto3 support.")
        return boto3.client(service_name, region_name=self.region)

    def _wait_for_job(self, transcribe, job_name: str):
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            job = transcribe.get_transcription_job(TranscriptionJobName=job_name)["TranscriptionJob"]
            status = job["TranscriptionJobStatus"]
            if status == "COMPLETED":
                return job
            if status == "FAILED":
                reason = job.get("FailureReason", "unknown reason")
                raise RuntimeError(f"Amazon Transcribe job {job_name} failed: {reason}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Amazon Transcribe job {job_name} did not finish in time.")
            time.sleep(self.poll_interval)

    def _load_transcript_json(self, s3, job, output_key):
        if self.output_bucket and output_key:
            obj = s3.get_object(Bucket=self.output_bucket, Key=output_key)
            return json.loads(obj["Body"].read().decode("utf-8"))

        transcript_uri = job["Transcript"]["TranscriptFileUri"]
        response = requests.get(transcript_uri, timeout=30)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _extract_transcript_text(transcript_json):
        transcripts = transcript_json.get("results", {}).get("transcripts", [])
        if not transcripts:
            raise RuntimeError("Amazon Transcribe response did not include transcript text.")
        return transcripts[0].get("transcript", "")

    def _language_code(self, language):
        if not language:
            return self.default_language_code
        if "-" in language:
            return language
        return {
            "en": "en-US",
            "es": "es-US",
            "fr": "fr-FR",
            "de": "de-DE",
            "it": "it-IT",
            "pt": "pt-BR",
            "ja": "ja-JP",
            "ko": "ko-KR",
            "zh": "zh-CN",
        }.get(language.lower(), self.default_language_code)

    @staticmethod
    def _join_s3_key(prefix, filename):
        clean_prefix = (prefix or "").strip("/")
        return f"{clean_prefix}/{filename}" if clean_prefix else filename

    @staticmethod
    def _job_name(audio_path: Path):
        stem = re.sub(r"[^A-Za-z0-9._-]+", "-", audio_path.stem).strip("-") or "audio"
        return f"sapat-{stem}-{uuid.uuid4().hex[:12]}"

    @staticmethod
    def _validate_audio_file(audio_file: str):
        if not os.path.exists(audio_file):
            raise ValueError(f"File {audio_file} does not exist.")

        valid_extensions = [".mp3", ".wav", ".flac", ".mp4", ".ogg", ".webm", ".amr"]
        if not any(str(audio_file).lower().endswith(ext) for ext in valid_extensions):
            raise ValueError(
                f"Unsupported audio file format: {audio_file}. Supported formats are {valid_extensions}."
            )

    @staticmethod
    def _media_format(audio_path: Path):
        return audio_path.suffix.lower().lstrip(".")
