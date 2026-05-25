import json
import os
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

from .base import TranscriptionBase


load_dotenv(".env")


SUPPORTED_MODEL_TYPES = {"ORACLE", "WHISPER_MEDIUM", "WHISPER_LARGE_V2"}
ORACLE_LANGUAGE_ALIASES = {
    "en": "en-US",
    "de": "de-DE",
    "es": "es-ES",
    "fr": "fr-FR",
    "it": "it-IT",
    "pt": "pt-BR",
}


class OracleSpeechTranscription(TranscriptionBase):
    """
    Oracle Cloud AI Speech transcription using Object Storage input/output.
    """

    def __init__(self, temperature: float):
        self.temperature = temperature
        self.config_file = os.getenv("OCI_CONFIG_FILE")
        self.profile = os.getenv("OCI_PROFILE", "DEFAULT")
        self.compartment_id = os.getenv("OCI_COMPARTMENT_ID")
        self.namespace = os.getenv("OCI_OBJECT_STORAGE_NAMESPACE")
        self.input_bucket = os.getenv("OCI_SPEECH_INPUT_BUCKET")
        self.output_bucket = os.getenv("OCI_SPEECH_OUTPUT_BUCKET", self.input_bucket or "")
        self.output_prefix = os.getenv("OCI_SPEECH_OUTPUT_PREFIX", "sapat-transcripts")
        self.model_type = os.getenv("OCI_SPEECH_MODEL_TYPE", "ORACLE").strip().upper()
        self.language_code = os.getenv("OCI_SPEECH_LANGUAGE_CODE", "en-US")
        self.poll_interval = float(os.getenv("OCI_SPEECH_POLL_INTERVAL_SECONDS", "5"))
        self.wait_seconds = float(os.getenv("OCI_SPEECH_WAIT_SECONDS", "900"))
        self.cleanup_input = os.getenv("OCI_SPEECH_CLEANUP_INPUT", "true").lower() != "false"

    def transcribe_audio(self, audio_file: str, **kwargs):
        audio_path = Path(audio_file)
        if not audio_path.exists():
            raise ValueError(f"File {audio_file} does not exist.")
        self._validate_configuration()

        oci = self._load_oci()
        config = self._load_config(oci)
        object_client = oci.object_storage.ObjectStorageClient(config)
        speech_client = oci.ai_speech.AIServiceSpeechClient(config)

        object_name = self._upload_audio(object_client, audio_path)
        try:
            job_id = self._create_job(oci, speech_client, object_name, kwargs)
            task = self._wait_for_task(speech_client, job_id)
            return self._read_transcript(object_client, task)
        finally:
            if self.cleanup_input:
                try:
                    object_client.delete_object(self.namespace, self.input_bucket, object_name)
                except Exception:
                    pass

    def _validate_configuration(self):
        required = {
            "OCI_COMPARTMENT_ID": self.compartment_id,
            "OCI_OBJECT_STORAGE_NAMESPACE": self.namespace,
            "OCI_SPEECH_INPUT_BUCKET": self.input_bucket,
            "OCI_SPEECH_OUTPUT_BUCKET": self.output_bucket,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"Missing Oracle Speech configuration: {', '.join(missing)}")

    @staticmethod
    def _load_oci():
        try:
            import oci
        except ImportError as exc:
            raise RuntimeError('Oracle Speech support requires installing sapat with "sapat[oracle]".') from exc
        return oci

    def _load_config(self, oci):
        if self.config_file:
            return oci.config.from_file(file_location=self.config_file, profile_name=self.profile)
        return oci.config.from_file(profile_name=self.profile)

    def _upload_audio(self, object_client, audio_path: Path) -> str:
        object_name = f"sapat-input/{uuid.uuid4()}-{audio_path.name}"
        with audio_path.open("rb") as audio:
            object_client.put_object(self.namespace, self.input_bucket, object_name, audio)
        return object_name

    def _create_job(self, oci, speech_client, object_name: str, kwargs) -> str:
        models = oci.ai_speech.models
        language = self._normalize_language(kwargs.get("language") or self.language_code)
        details = models.CreateTranscriptionJobDetails(
            compartment_id=self.compartment_id,
            display_name=f"sapat-{uuid.uuid4()}",
            input_location=models.ObjectListInlineInputLocation(
                object_locations=[
                    models.ObjectLocation(
                        namespace_name=self.namespace,
                        bucket_name=self.input_bucket,
                        object_names=[object_name],
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
                language_code=language,
            ),
        )
        response = speech_client.create_transcription_job(details)
        job = response.data
        job_id = getattr(job, "id", None)
        if not job_id:
            raise RuntimeError("Oracle Speech did not return a transcription job id.")
        return job_id

    def _wait_for_task(self, speech_client, job_id: str):
        deadline = time.time() + self.wait_seconds
        last_state = "UNKNOWN"
        while time.time() < deadline:
            response = speech_client.list_transcription_tasks(job_id)
            tasks = list(getattr(response.data, "items", None) or getattr(response.data, "data", None) or [])
            if tasks:
                task = tasks[0]
                state = str(getattr(task, "lifecycle_state", "") or "").upper()
                if state == "SUCCEEDED":
                    task_id = getattr(task, "id", None)
                    if not task_id:
                        raise RuntimeError("Oracle Speech task succeeded without a task id.")
                    return speech_client.get_transcription_task(job_id, task_id).data
                if state in {"FAILED", "CANCELED"}:
                    details = getattr(task, "lifecycle_details", "") or state
                    raise RuntimeError(f"Oracle Speech task failed: {details}")
                last_state = state or last_state
            time.sleep(self.poll_interval)
        raise TimeoutError(f"Oracle Speech transcription timed out while task was {last_state}.")

    def _normalize_language(self, language: str) -> str:
        if self.model_type not in SUPPORTED_MODEL_TYPES:
            supported = ", ".join(sorted(SUPPORTED_MODEL_TYPES))
            raise ValueError(f"Unsupported OCI_SPEECH_MODEL_TYPE {self.model_type!r}; expected one of {supported}.")

        if not language:
            return self.language_code

        language = ORACLE_LANGUAGE_ALIASES.get(language.lower(), language)
        if self.model_type == "ORACLE":
            if language.lower() == "auto":
                raise ValueError("Oracle Speech model ORACLE requires an explicit locale such as en-US.")
            if "-" not in language:
                raise ValueError(
                    "Oracle Speech model ORACLE requires a locale code such as en-US. "
                    "Set OCI_SPEECH_MODEL_TYPE=WHISPER_MEDIUM or WHISPER_LARGE_V2 for auto/short-code use."
                )
        return language

    def _read_transcript(self, object_client, task) -> str:
        output_location = getattr(task, "output_location", None)
        if not output_location:
            raise RuntimeError("Oracle Speech task did not include an output location.")
        object_names = list(getattr(output_location, "object_names", None) or [])
        if not object_names:
            raise RuntimeError("Oracle Speech task did not include transcript object names.")
        response = object_client.get_object(
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
