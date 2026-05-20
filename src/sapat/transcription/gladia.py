import os
import time

import requests
from dotenv import load_dotenv

from .base import TranscriptionBase


load_dotenv(".env")


class GladiaTranscription(TranscriptionBase):
    """
    Gladia API implementation for pre-recorded transcription.
    """

    def __init__(self, temperature: float, poll_interval: float = None, timeout: float = None):
        self.api_key = os.getenv("GLADIA_API_KEY")
        self.upload_endpoint = os.getenv("GLADIA_UPLOAD_ENDPOINT", "https://api.gladia.io/v2/upload")
        self.transcription_endpoint = os.getenv(
            "GLADIA_TRANSCRIPTION_ENDPOINT",
            "https://api.gladia.io/v2/pre-recorded",
        )
        self.temperature = temperature
        self.poll_interval = poll_interval or float(os.getenv("GLADIA_POLL_INTERVAL_SECONDS", "5"))
        self.timeout = timeout or float(os.getenv("GLADIA_TIMEOUT_SECONDS", "900"))

    def transcribe_audio(self, audio_file: str, **kwargs):
        """
        Transcribes an audio file using Gladia's pre-recorded API.

        Parameters:
        - audio_file (str): Path to the audio file.
        - kwargs: Additional parameters for transcription.

        Returns:
        - str: The full transcript text.
        """
        self._validate_audio_file(audio_file)

        audio_url = self._upload_audio(audio_file)
        result_url = self._create_transcription(audio_url, language=kwargs.get("language"))
        result = self._poll_result(result_url)
        return self._extract_transcript(result)

    def _upload_audio(self, audio_file: str):
        with open(audio_file, "rb") as f:
            response = requests.post(
                self.upload_endpoint,
                headers=self._headers(),
                files={"audio": f},
            )

        self._raise_for_response(response, "Upload failed")
        data = response.json()
        audio_url = data.get("audio_url")
        if not audio_url:
            raise Exception("Gladia upload response did not include audio_url.")
        return audio_url

    def _create_transcription(self, audio_url: str, language: str = None):
        payload = {
            "audio_url": audio_url,
            "language_config": {
                "languages": [language] if language else [],
                "code_switching": False,
            },
        }

        response = requests.post(
            self.transcription_endpoint,
            headers=self._headers(content_type="application/json"),
            json=payload,
        )

        self._raise_for_response(response, "Transcription job creation failed")
        data = response.json()
        result_url = data.get("result_url")
        job_id = data.get("id")

        if result_url:
            return result_url
        if job_id:
            return f"{self.transcription_endpoint.rstrip('/')}/{job_id}"

        raise Exception("Gladia transcription response did not include result_url or id.")

    def _poll_result(self, result_url: str):
        deadline = time.monotonic() + self.timeout

        while True:
            response = requests.get(result_url, headers=self._headers())
            self._raise_for_response(response, "Transcription result polling failed")
            data = response.json()
            status = data.get("status")

            if status == "done":
                return data
            if status == "error":
                error = data.get("error_code") or data.get("error") or "unknown error"
                raise Exception(f"Gladia transcription failed: {error}")
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for Gladia transcription to finish.")

            time.sleep(self.poll_interval)

    def _extract_transcript(self, result):
        transcription = result.get("result", {}).get("transcription", {})
        full_transcript = transcription.get("full_transcript")
        if full_transcript is not None:
            return full_transcript

        utterances = transcription.get("utterances") or []
        if utterances:
            return "\n".join(utterance.get("text", "") for utterance in utterances).strip()

        raise Exception("Gladia result did not include a transcript.")

    def _headers(self, content_type: str = None):
        if not self.api_key:
            raise ValueError("GLADIA_API_KEY is required for Gladia transcription.")

        headers = {"x-gladia-key": self.api_key}
        if content_type:
            headers["Content-Type"] = content_type
        return headers

    def _raise_for_response(self, response, message: str):
        if response.ok:
            return
        raise Exception(f"{message}: {response.status_code} {response.text}")

    def _validate_audio_file(self, audio_file: str):
        if not os.path.exists(audio_file):
            raise ValueError(f"File {audio_file} does not exist.")

        valid_extensions = [".mp3", ".wav", ".flac", ".m4a", ".ogg"]
        if not any(str(audio_file).lower().endswith(ext) for ext in valid_extensions):
            raise ValueError(
                f"Unsupported audio file format: {audio_file}. Supported formats are {valid_extensions}."
            )
