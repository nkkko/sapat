import os
import time
from typing import Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI

from .base import TranscriptionBase

load_dotenv(".env")

DEFAULT_ASSEMBLYAI_ENDPOINT = "https://api.assemblyai.com/v2"
SUPPORTED_AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac")


class AssemblyAITranscription(TranscriptionBase):
    """
    AssemblyAI REST API implementation for transcription.
    """

    def __init__(
        self,
        temperature: float,
        response_format: str = "json",
        poll_interval_seconds: Optional[float] = None,
        timeout_seconds: Optional[float] = None,
    ):
        self.api_key = os.getenv("ASSEMBLYAI_API_KEY")
        self.endpoint = os.getenv(
            "ASSEMBLYAI_API_ENDPOINT",
            DEFAULT_ASSEMBLYAI_ENDPOINT,
        ).rstrip("/")
        self.model_name_chat = os.getenv("OPENAI_MODEL_NAME_CHAT")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.temperature = temperature
        self.response_format = response_format
        self.poll_interval_seconds = (
            poll_interval_seconds
            if poll_interval_seconds is not None
            else float(os.getenv("ASSEMBLYAI_POLL_INTERVAL_SECONDS", "3"))
        )
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else float(os.getenv("ASSEMBLYAI_TIMEOUT_SECONDS", "600"))
        )
        self.max_file_size_mb = 25

    def transcribe_audio(self, audio_file: str, **kwargs):
        """
        Uploads an audio file to AssemblyAI, submits a transcript job, and polls
        until the transcript is ready.
        """
        self._validate_audio_file(audio_file)
        upload_url = self._upload_audio(audio_file)
        transcript_id = self._submit_transcript(upload_url, **kwargs)
        transcript = self._poll_transcript(transcript_id)

        if self.response_format == "text":
            return transcript.get("text", "")
        return {
            "text": transcript.get("text", ""),
            "id": transcript_id,
            "status": transcript.get("status"),
        }

    def generate_corrected_transcript(
        self,
        audio_file: str,
        temperature: float,
        system_prompt: str,
    ):
        """
        Uses OpenAI chat completion settings, when configured, to clean up the
        AssemblyAI transcript.
        """
        if not self.openai_api_key or not self.model_name_chat:
            raise ValueError(
                "Correction for AssemblyAI requires OPENAI_API_KEY and "
                "OPENAI_MODEL_NAME_CHAT."
            )

        transcription = self.transcribe_audio(audio_file)
        transcription_text = (
            transcription.get("text", "")
            if isinstance(transcription, dict)
            else transcription
        )

        client = OpenAI(api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model=self.model_name_chat,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcription_text},
            ],
        )
        return response.choices[0].message.content

    def _upload_audio(self, audio_file: str):
        with open(audio_file, "rb") as f:
            response = requests.post(
                f"{self.endpoint}/upload",
                headers=self._headers(),
                data=f,
            )
        self._raise_for_response(response, "AssemblyAI upload failed")
        return response.json()["upload_url"]

    def _submit_transcript(self, audio_url: str, **kwargs):
        payload = {
            "audio_url": audio_url,
        }
        language = kwargs.get("language")
        if language:
            payload["language_code"] = language

        response = requests.post(
            f"{self.endpoint}/transcript",
            headers={**self._headers(), "Content-Type": "application/json"},
            json=payload,
        )
        self._raise_for_response(response, "AssemblyAI transcript submission failed")
        return response.json()["id"]

    def _poll_transcript(self, transcript_id: str):
        deadline = time.time() + self.timeout_seconds
        while time.time() < deadline:
            response = requests.get(
                f"{self.endpoint}/transcript/{transcript_id}",
                headers=self._headers(),
            )
            self._raise_for_response(response, "AssemblyAI transcript polling failed")
            transcript = response.json()
            status = transcript.get("status")
            if status == "completed":
                return transcript
            if status == "error":
                raise RuntimeError(
                    f"AssemblyAI transcription failed: {transcript.get('error')}"
                )
            time.sleep(self.poll_interval_seconds)

        raise TimeoutError(
            f"AssemblyAI transcription timed out after {self.timeout_seconds} seconds."
        )

    def _headers(self):
        if not self.api_key:
            raise ValueError(
                "ASSEMBLYAI_API_KEY is required for AssemblyAI transcription."
            )
        return {"Authorization": self.api_key}

    def _raise_for_response(self, response, message: str):
        if response.status_code >= 400:
            raise RuntimeError(f"{message} ({response.status_code}): {response.text}")

    def _validate_audio_file(self, audio_file: str):
        if not os.path.exists(audio_file):
            raise ValueError(f"File {audio_file} does not exist.")

        file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            raise ValueError(
                f"File size exceeds the maximum limit of {self.max_file_size_mb} MB."
            )

        if not str(audio_file).lower().endswith(SUPPORTED_AUDIO_EXTENSIONS):
            raise ValueError(
                f"Unsupported audio file format: {audio_file}. "
                f"Supported formats are {list(SUPPORTED_AUDIO_EXTENSIONS)}."
            )
