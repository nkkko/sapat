# ABOUTME: xAI (Grok) transcription provider
# ABOUTME: Uses xAI's OpenAI-compatible audio transcriptions endpoint for speech-to-text

from sapat.providers import register
from sapat.providers.base import AudioFormat, ProviderConfig
from sapat.providers.openai_compat import OpenAICompatProvider


@register
class XAIProvider(OpenAICompatProvider):
    name = "xai"
    config = ProviderConfig(
        required_env_vars=["XAI_API_KEY"],
        default_model="whisper-large-v3",
    )
    base_url = "https://api.x.ai/v1/audio/transcriptions"
    _env_key_for_auth = "XAI_API_KEY"
