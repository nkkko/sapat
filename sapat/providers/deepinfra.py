# ABOUTME: DeepInfra transcription provider
# ABOUTME: Uses DeepInfra's OpenAI-compatible Whisper endpoint for speech-to-text

from sapat.providers import register
from sapat.providers.base import AudioFormat, ProviderConfig
from sapat.providers.openai_compat import OpenAICompatProvider


@register
class DeepInfraProvider(OpenAICompatProvider):
    name = "deepinfra"
    config = ProviderConfig(
        required_env_vars=["DEEPINFRA_API_KEY"],
        default_model="openai/whisper-large-v3",
    )
    base_url = "https://api.deepinfra.com/v1/openai/audio/transcriptions"
    _env_key_for_auth = "DEEPINFRA_API_KEY"
