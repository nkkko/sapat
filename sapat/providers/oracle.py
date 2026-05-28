# ABOUTME: Oracle Cloud AI Speech transcription provider
# ABOUTME: Uploads audio to Object Storage, creates job, polls task, reads transcript

import json
import os
import time
import uuid
from pathlib import Path
from typing import Optional

from sapat.providers import register
from sapat.providers.async_poll import AsyncPollProvider
from sapat.providers.base import (
    ProviderConfig,
    TranscriptionResult,
)

SUPPORTED_MODEL_TYPES = {"ORACLE", "WHISPER_MEDIUM", "WHISPER_LARGE_V2"}

ORACLE_LANGUAGE_ALIASES = {
    "en": "en-US",
    "de": "de-DE",
    "es": "es-ES",
    "fr": "fr-FR",
    "it": "it-IT",
    "pt": "pt-BR",
}


@register
class OracleProvider(AsyncPollProvider):
    """Oracle Cloud AI Speech transcription via Object Storage."""

    name = "oracle"
    config = ProviderConfig(
        required_env_vars=["ORACLE_COMPARTMENT_ID"],
        required_packages=["oci"],
        extras_key="oracle",
        default_model="default",
    )

    poll_interval: float = float(os.getenv("OCI_SPEECH_POLL_INTERVAL_SECONDS", "5"))
    max_poll_time: float = float(os.getenv("OCI_SPEECH_WAIT_SECONDS", "900"))

    def __init__(self):
        super().__init__()
        self.config_file = os.getenv("OCI_CONFIG_FILE")
        self.profile = os.getenv("OCI_PROFILE", "DEFAULT")
        self.compartment_id = os.getenv("OCI_COMPARTMENT_ID")
        self.namespace = os.getenv("OCI_OBJECT_STORAGE_NAMESPACE")
        self.input_bucket = os.getenv("OCI_SPEECH_INPUT_BUCKET")
        self.output_bucket = os.getenv(
            "OCI_SPEECH_OUTPUT_BUCKET", self.input_bucket or ""
        )
        self.output_prefix = os.getenv(
            "OCI_SPEECH_OUTPUT_PREFIX", "sapat-transcripts"
        )
        self.model_type = os.getenv(
            "OCI_SPEECH_MODEL_TYPE", "ORACLE"
        ).strip().upper()
        self.language_code = os.getenv("OCI_SPEECH_LANGUAGE_CODE", "en-US")
        self.cleanup_input = (
            os.getenv("OCI_SPEECH_CLEANUP_INPUT", "true").lower() != "false"
        )
        # Stashed OCI clients and state during _upload for use in _poll/_fetch_result
        self._object_client = None
        self._speech_client = None
        self._object_name = None

    @classmethod
    def is_available(cls) -> bool:
        cfg = cls.config
        for var in cfg.required_env_vars:
            if not os.getenv(var):
                return False
        for pkg in cfg.required_packages:
            try:
                __import__(pkg)
            except ImportError:
                return False
        return True

    def _load_oci(self):
        try:
            import oci
        except ImportError as exc:
            raise RuntimeError(
                'Oracle Speech support requires installing sapat with "sapat[oracle]".'
            ) from exc
        return oci

    def _load_config(self, oci):
        if self.config_file:
            return oci.config.from_file(
                file_location=self.config_file, profile_name=self.profile
            )
        return oci.config.from_file(profile_name=self.profile)

    def _normalize_language(self, language: str) -> str:
        if self.model_type not in SUPPORTED_MODEL_TYPES:
            supported = ", ".join(sorted(SUPPORTED_MODEL_TYPES))
            raise ValueError(
                f"Unsupported OCI_SPEECH_MODEL_TYPE {self.model_type!r}; "
                f"expected one of {supported}."
            )

        if not language:
            return self.language_code

        language = ORACLE_LANGUAGE_ALIASES.get(language.lower(), language)
        if self.model_type == "ORACLE":
            if language.lower() == "auto":
                raise ValueError(
                    "Oracle Speech model ORACLE requires an explicit locale such as en-US."
                )
            if "-" not in language:
                raise ValueError(
                    "Oracle Speech model ORACLE requires a locale code such as en-US. "
                    "Set OCI_SPEECH_MODEL_TYPE=WHISPER_MEDIUM or WHISPER_LARGE_V2 "
                    "for auto/short-code use."
                )
        return language

    def _upload(self, audio_file: str, model: str, language: str, **kwargs) -> str:
        self._validate_configuration()

        oci = self._load_oci()
        config = self._load_config(oci)
        self._object_client = oci.object_storage.ObjectStorageClient(config)
        self._speech_client = oci.ai_speech.AIServiceSpeechClient(config)

        audio_path = Path(audio_file)
        self._object_name = self._upload_audio(audio_path)
        return self._create_job(oci, language)

    def _validate_configuration(self) -> None:
        required = {
            "OCI_COMPARTMENT_ID": self.compartment_id,
            "OCI_OBJECT_STORAGE_NAMESPACE": self.namespace,
            "OCI_SPEECH_INPUT_BUCKET": self.input_bucket,
            "OCI_SPEECH_OUTPUT_BUCKET": self.output_bucket,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(
                f"Missing Oracle Speech configuration: {', '.join(missing)}"
            )

    def _upload_audio(self, audio_path: Path) -> str:
        object_name = f"sapat-input/{uuid.uuid4()}-{audio_path.name}"
        with audio_path.open("rb") as audio:
            self._object_client.put_object(
                self.namespace, self.input_bucket, object_name, audio
            )
        return object_name

    def _create_job(self, oci, language: str) -> str:
        models = oci.ai_speech.models
        normalized = self._normalize_language(language)
        details = models.CreateTranscriptionJobDetails(
            compartment_id=self.compartment_id,
            display_name=f"sapat-{uuid.uuid4()}",
            input_location=models.ObjectListInlineInputLocation(
                object_locations=[
                    models.ObjectLocation(
                        namespace_name=self.namespace,
                        bucket_name=self.input_bucket,
                        object_names=[self._object_name],
                    )
                ]
            ),
            output_location=models.OutputLocation(
                namespace_name=self.namespace,
                bucket_name=self.output_bucket,
                prefix=self.output_prefix,
            ),
            model_details=models.TranscriptionModelDetails(
                model_type=self.model_type,
                domain="GENERIC",
                language_code=normalized,
            ),
        )
        response = self._speech_client.create_transcription_job(details)
        job_id = getattr(response.data, "id", None)
        if not job_id:
            raise RuntimeError(
                "Oracle Speech did not return a transcription job id."
            )
        return job_id

    def _poll(self, job_id: str) -> str:
        response = self._speech_client.list_transcription_tasks(job_id)
        tasks = list(
            getattr(response.data, "items", None)
            or getattr(response.data, "data", None)
            or []
        )
        if not tasks:
            return "pending"

        task = tasks[0]
        state = str(getattr(task, "lifecycle_state", "") or "").upper()
        if state == "SUCCEEDED":
            return "completed"
        if state in {"FAILED", "CANCELED"}:
            return "failed"
        return "pending"

    def _fetch_result(self, job_id: str) -> TranscriptionResult:
        # Get the task details to find the output location
        response = self._speech_client.list_transcription_tasks(job_id)
        tasks = list(
            getattr(response.data, "items", None)
            or getattr(response.data, "data", None)
            or []
        )
        if not tasks:
            raise RuntimeError("Oracle Speech: no tasks found for completed job.")

        task = tasks[0]
        task_id = getattr(task, "id", None)
        if not task_id:
            raise RuntimeError("Oracle Speech task succeeded without a task id.")

        task_detail = self._speech_client.get_transcription_task(
            job_id, task_id
        ).data
        text = self._read_transcript(task_detail)

        # Cleanup input object if configured
        if self.cleanup_input and self._object_name:
            try:
                self._object_client.delete_object(
                    self.namespace, self.input_bucket, self._object_name
                )
            except Exception:
                pass

        return TranscriptionResult(text=text)

    def _read_transcript(self, task) -> str:
        output_location = getattr(task, "output_location", None)
        if not output_location:
            raise RuntimeError(
                "Oracle Speech task did not include an output location."
            )
        object_names = list(
            getattr(output_location, "object_names", None) or []
        )
        if not object_names:
            raise RuntimeError(
                "Oracle Speech task did not include transcript object names."
            )
        response = self._object_client.get_object(
            getattr(output_location, "namespace_name", self.namespace),
            getattr(output_location, "bucket_name", self.output_bucket),
            object_names[0],
        )
        raw = response.data.content.decode("utf-8")
        return self._extract_text(raw)

    @staticmethod
    def _extract_text(raw: str) -> str:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return raw.strip()

        for key in ("transcription", "text"):
            value = payload.get(key)
            if isinstance(value, str):
                return value.strip()

        transcriptions = payload.get("transcriptions")
        if isinstance(transcriptions, list):
            parts = []
            for item in transcriptions:
                if isinstance(item, dict):
                    value = item.get("transcription") or item.get("text")
                    if isinstance(value, str):
                        parts.append(value.strip())
            if parts:
                return "\n".join(part for part in parts if part)

        return json.dumps(payload, indent=2)
