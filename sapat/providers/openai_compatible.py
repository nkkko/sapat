# ABOUTME: Generic OpenAI-compatible transcription provider
# ABOUTME: Lets users point Sapat at any compatible /audio/transcriptions endpoint

import os

from sapat.providers import register
from sapat.providers.base import AudioFormat, ProviderConfig
from sapat.providers.openai_compat import OpenAICompatProvider


@register
class OpenAICompatibleProvider(OpenAICompatProvider):
    name = "openai_compatible"
    config = ProviderConfig(
        required_env_vars=[
            "OPENAI_COMPAT_STT_BASE_URL",
            "OPENAI_COMPAT_STT_API_KEY",
        ],
        max_file_size_mb=25.0,
        preferred_format=AudioFormat.MP3,
        supports_correction=False,
        default_model="default",
    )

    def __init__(self):
        super().__init__()
        self.base_url = self._resolve_base_url()
        self._auth_header_name = os.getenv(
            "OPENAI_COMPAT_STT_AUTH_HEADER",
            "Authorization",
        )
        self._auth_header_prefix = os.getenv(
            "OPENAI_COMPAT_STT_AUTH_PREFIX",
            "Bearer ",
        )
        self._env_key_for_auth = "OPENAI_COMPAT_STT_API_KEY"

    @classmethod
    def is_available(cls) -> bool:
        return bool(
            os.getenv("OPENAI_COMPAT_STT_BASE_URL")
            and os.getenv("OPENAI_COMPAT_STT_API_KEY")
        )

    def resolve_model(self, model_alias: str) -> str:
        if model_alias == "default":
            return os.getenv("OPENAI_COMPAT_STT_MODEL", "whisper-1")
        return model_alias

    @staticmethod
    def _resolve_base_url() -> str:
        base_url = os.getenv("OPENAI_COMPAT_STT_BASE_URL", "").strip()
        if not base_url:
            raise ValueError("OPENAI_COMPAT_STT_BASE_URL must be set")

        base_url = base_url.rstrip("/")
        if base_url.endswith("/audio/transcriptions"):
            return base_url
        if base_url.endswith("/v1"):
            return f"{base_url}/audio/transcriptions"
        return f"{base_url}/v1/audio/transcriptions"
