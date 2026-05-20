import json
import os
import re
import time
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import boto3
import requests
from dotenv import load_dotenv

from .base import TranscriptionBase

# Load environment variables
load_dotenv(".env")


class AWSTranscribeTranscription(TranscriptionBase):
    """
    Amazon Transcribe implementation for batch transcription jobs.
    """

    LANGUAGE_CODE_MAP = {
        "en": "en-US",
        "es": "es-US",
        "fr": "fr-FR",
        "de": "de-DE",
        "it": "it-IT",
        "pt": "pt-BR",
        "ja": "ja-JP",
        "ko": "ko-KR",
        "zh": "zh-CN",
    }

    def __init__(self, temperature: float):
        """
        Initializes the AWS Transcribe provider.

        Parameters:
        - temperature (float): Kept for CLI parity with the other providers.
        """
        self.region = (
            os.getenv("AWS_TRANSCRIBE_REGION")
            or os.getenv("AWS_DEFAULT_REGION")
            or os.getenv("AWS_REGION")
            or "us-east-1"
        )
        self.bucket = os.getenv("AWS_TRANSCRIBE_S3_BUCKET")
        self.prefix = os.getenv("AWS_TRANSCRIBE_S3_PREFIX", "sapat-transcribe").strip("/")
        self.output_bucket = os.getenv("AWS_TRANSCRIBE_OUTPUT_BUCKET")
        self.output_prefix = os.getenv(
            "AWS_TRANSCRIBE_OUTPUT_PREFIX",
            "sapat-transcripts",
        ).strip("/")
        self.language_code = os.getenv("AWS_TRANSCRIBE_LANGUAGE_CODE")
        self.identify_language = (
            os.getenv("AWS_TRANSCRIBE_IDENTIFY_LANGUAGE", "false").lower() == "true"
        )
        self.poll_seconds = float(os.getenv("AWS_TRANSCRIBE_POLL_SECONDS", "5"))
        self.max_wait_seconds = float(
            os.getenv("AWS_TRANSCRIBE_MAX_WAIT_SECONDS", "900")
        )
        self.delete_uploaded_media = (
            os.getenv("AWS_TRANSCRIBE_DELETE_UPLOADED_MEDIA", "false").lower() == "true"
        )
        self.max_file_size_mb = int(os.getenv("AWS_TRANSCRIBE_MAX_FILE_SIZE_MB", "100"))
        self.temperature = temperature
        self.s3 = boto3.client("s3", region_name=self.region)
        self.transcribe = boto3.client("transcribe", region_name=self.region)

    def transcribe_audio(self, audio_file: str, **kwargs):
        """
        Transcribes an audio file using Amazon Transcribe batch jobs.

        Parameters:
        - audio_file (str): Path to the audio file.
        - kwargs: Additional parameters for transcription.

        Returns:
        - str: The transcript text returned by Amazon Transcribe.
        """
        self._validate_configuration()
        self._validate_audio_file(audio_file)

        job_name = self._job_name_for(audio_file)
        media_key = self._media_key_for(job_name, audio_file)
        self.s3.upload_file(audio_file, self.bucket, media_key)

        try:
            self._start_job(job_name, media_key, audio_file, kwargs.get("language"))
            transcript_uri = self._wait_for_transcript(job_name)
            payload = self._load_transcript_payload(transcript_uri)
            return self._extract_transcript(payload)
        finally:
            if self.delete_uploaded_media:
                self.s3.delete_object(Bucket=self.bucket, Key=media_key)

    def _validate_configuration(self):
        if not self.bucket:
            raise ValueError("AWS_TRANSCRIBE_S3_BUCKET is required for AWS Transcribe.")

    def _validate_audio_file(self, audio_file: str):
        if not os.path.exists(audio_file):
            raise ValueError(f"File {audio_file} does not exist.")

        file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            raise Exception(
                f"File size exceeds the maximum limit of {self.max_file_size_mb} MB."
            )

        valid_extensions = [".mp3", ".mp4", ".wav", ".flac", ".m4a"]
        if Path(audio_file).suffix.lower() not in valid_extensions:
            raise ValueError(
                f"Unsupported audio file format: {audio_file}. "
                f"Supported formats are {valid_extensions}."
            )

    def _start_job(self, job_name: str, media_key: str, audio_file: str, language: str):
        media_format = Path(audio_file).suffix.lower().lstrip(".")
        args = {
            "TranscriptionJobName": job_name,
            "Media": {"MediaFileUri": f"s3://{self.bucket}/{media_key}"},
            "MediaFormat": media_format,
        }

        if self.identify_language:
            args["IdentifyLanguage"] = True
        else:
            args["LanguageCode"] = self._language_code_for(language)

        if self.output_bucket:
            args["OutputBucketName"] = self.output_bucket
            args["OutputKey"] = self._output_key_for(job_name)

        self.transcribe.start_transcription_job(**args)

    def _wait_for_transcript(self, job_name: str):
        deadline = time.time() + self.max_wait_seconds

        while True:
            response = self.transcribe.get_transcription_job(TranscriptionJobName=job_name)
            job = response["TranscriptionJob"]
            status = job["TranscriptionJobStatus"]

            if status == "COMPLETED":
                return job["Transcript"]["TranscriptFileUri"]

            if status == "FAILED":
                reason = job.get("FailureReason", "unknown reason")
                raise Exception(f"AWS Transcribe job failed: {reason}")

            if time.time() >= deadline:
                raise TimeoutError(f"AWS Transcribe job timed out: {job_name}")

            time.sleep(self.poll_seconds)

    def _load_transcript_payload(self, transcript_uri: str):
        if transcript_uri.startswith("s3://"):
            bucket, key = self._parse_s3_uri(transcript_uri)
            response = self.s3.get_object(Bucket=bucket, Key=key)
            return json.loads(response["Body"].read().decode("utf-8"))

        response = requests.get(transcript_uri)
        if response.status_code != 200:
            raise Exception(f"Unable to download AWS transcript: {response.text}")
        return response.json()

    @classmethod
    def _extract_transcript(cls, payload):
        transcripts = payload.get("results", {}).get("transcripts", [])
        if not transcripts:
            return ""
        return transcripts[0].get("transcript", "")

    def _language_code_for(self, language: str):
        if self.language_code:
            return self.language_code

        if not language:
            return "en-US"

        normalized = language.strip()
        if "-" in normalized:
            return normalized

        return self.LANGUAGE_CODE_MAP.get(normalized.lower(), normalized)

    def _media_key_for(self, job_name: str, audio_file: str):
        suffix = Path(audio_file).suffix.lower()
        return f"{self.prefix}/{job_name}{suffix}" if self.prefix else f"{job_name}{suffix}"

    def _output_key_for(self, job_name: str):
        return f"{self.output_prefix}/{job_name}.json" if self.output_prefix else f"{job_name}.json"

    @staticmethod
    def _job_name_for(audio_file: str):
        stem = (
            re.sub(r"[^0-9A-Za-z._-]+", "-", Path(audio_file).stem).strip("-._")
            or "media"
        )
        return f"sapat-{stem}-{uuid4().hex[:10]}"

    @staticmethod
    def _parse_s3_uri(uri: str):
        parsed = urlparse(uri)
        return parsed.netloc, parsed.path.lstrip("/")
