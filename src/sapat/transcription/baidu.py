import base64
import os
import subprocess
import uuid

import requests
from dotenv import load_dotenv

from .base import TranscriptionBase


load_dotenv(".env")


class BaiduTranscription(TranscriptionBase):
    """
    Baidu Speech Recognition API implementation for transcription.
    """

    TOKEN_ENDPOINT = "https://aip.baidubce.com/oauth/2.0/token"
    TRANSCRIPTION_ENDPOINT = "https://vop.baidu.com/server_api"
    LANGUAGE_DEV_PIDS = {
        "zh": 1537,
        "zh-cn": 1537,
        "zh_cn": 1537,
        "cmn": 1537,
        "en": 1737,
        "en-us": 1737,
        "en_us": 1737,
    }

    def __init__(self, temperature: float, response_format: str = "json"):
        """
        Initializes the BaiduTranscription class.

        Parameters:
        - temperature (float): Kept for compatibility with the shared CLI.
        - response_format (str): Kept for compatibility with other providers.
        """
        self.api_key = os.getenv("BAIDU_API_KEY")
        self.secret_key = os.getenv("BAIDU_SECRET_KEY")
        self.cuid = os.getenv("BAIDU_CUID") or f"sapat-{uuid.getnode()}"
        self.dev_pid = os.getenv("BAIDU_DEV_PID")
        self.audio_format = os.getenv("BAIDU_AUDIO_FORMAT", "mp3")
        self.sample_rate = int(os.getenv("BAIDU_SAMPLE_RATE", "16000"))
        self.endpoint = os.getenv("BAIDU_API_ENDPOINT", self.TRANSCRIPTION_ENDPOINT)
        self.token_endpoint = os.getenv("BAIDU_TOKEN_ENDPOINT", self.TOKEN_ENDPOINT)
        self.temperature = temperature
        self.response_format = response_format
        self.max_file_size_mb = int(os.getenv("BAIDU_MAX_FILE_SIZE_MB", "10"))

    def transcribe_audio(self, audio_file: str, **kwargs):
        """
        Transcribes an audio file using Baidu's short speech recognition API.

        Parameters:
        - audio_file (str): Path to the audio file.
        - kwargs: Additional parameters for transcription.

        Returns:
        - dict: The transcription result with a text field.
        """
        self._validate_audio_file(audio_file)

        with open(audio_file, "rb") as f:
            audio_data = f.read()

        payload = {
            "format": kwargs.get("format", self.audio_format),
            "rate": int(kwargs.get("rate", self.sample_rate)),
            "channel": int(kwargs.get("channel", 1)),
            "cuid": kwargs.get("cuid", self.cuid),
            "token": self._get_access_token(),
            "len": len(audio_data),
            "speech": base64.b64encode(audio_data).decode("utf-8"),
            "dev_pid": self._resolve_dev_pid(kwargs.get("language")),
        }

        response = requests.post(
            self.endpoint,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        result = response.json()

        if response.status_code == 200 and result.get("err_no") == 0:
            transcript = result.get("result", "")
            if isinstance(transcript, list):
                transcript = " ".join(transcript)
            return {"text": transcript}

        error_message = result.get("err_msg", response.text)
        raise Exception(f"Baidu transcription failed: {error_message}")

    def _get_access_token(self):
        if not self.api_key or not self.secret_key:
            raise ValueError("BAIDU_API_KEY and BAIDU_SECRET_KEY must be set.")

        response = requests.get(
            self.token_endpoint,
            params={
                "grant_type": "client_credentials",
                "client_id": self.api_key,
                "client_secret": self.secret_key,
            },
            timeout=30,
        )
        result = response.json()

        if response.status_code == 200 and result.get("access_token"):
            return result["access_token"]

        error_message = result.get("error_description", response.text)
        raise Exception(f"Could not fetch Baidu access token: {error_message}")

    def _resolve_dev_pid(self, language):
        if self.dev_pid:
            return int(self.dev_pid)

        if not language:
            return self.LANGUAGE_DEV_PIDS["en"]

        normalized_language = language.lower()
        return self.LANGUAGE_DEV_PIDS.get(
            normalized_language,
            self.LANGUAGE_DEV_PIDS.get(normalized_language.split("-")[0], self.LANGUAGE_DEV_PIDS["en"]),
        )

    def _validate_audio_file(self, audio_file: str):
        if not os.path.exists(audio_file):
            raise ValueError(f"File {audio_file} does not exist.")

        file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            raise Exception(f"File size exceeds the maximum limit of {self.max_file_size_mb} MB.")

        valid_extensions = [".mp3", ".wav", ".pcm", ".amr", ".m4a"]
        if not any(str(audio_file).lower().endswith(ext) for ext in valid_extensions):
            raise ValueError(f"Unsupported audio file format: {audio_file}. Supported formats are {valid_extensions}.")

    @staticmethod
    def convert_to_mp3(input_file: str, output_file: str, quality: str):
        command = [
            "ffmpeg",
            "-i",
            input_file,
            "-vn",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-b:a",
            "64k",
            output_file,
        ]
        subprocess.run(command, check=True)
