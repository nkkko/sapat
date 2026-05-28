# ABOUTME: Venice transcription provider
# ABOUTME: Uses Venice AI's OpenAI-compatible Whisper endpoint for speech-to-text

from sapat.providers import register
from sapat.providers.base import AudioFormat, ProviderConfig
from sapat.providers.openai_compat import OpenAICompatProvider


@register
class VeniceProvider(OpenAICompatProvider):
    name = "venice"
    config = ProviderConfig(
        required_env_vars=["VENICE_API_KEY"],
        default_model="whisper-large-v3",
    )
    base_url = "https://api.venice.ai/api/v1/audio/transcriptions"
    _env_key_for_auth = "VENICE_API_KEY"
