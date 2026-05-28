# ABOUTME: Together AI transcription provider
# ABOUTME: Uses Together AI's OpenAI-compatible Whisper endpoint for speech-to-text

from sapat.providers import register
from sapat.providers.base import AudioFormat, ProviderConfig
from sapat.providers.openai_compat import OpenAICompatProvider


@register
class TogetherProvider(OpenAICompatProvider):
    name = "together"
    config = ProviderConfig(
        required_env_vars=["TOGETHER_API_KEY"],
        default_model="openai/whisper-large-v3",
    )
    base_url = "https://api.together.xyz/v1/audio/transcriptions"
    _env_key_for_auth = "TOGETHER_API_KEY"
