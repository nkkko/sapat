# ABOUTME: Lemonfox transcription provider
# ABOUTME: Uses Lemonfox's OpenAI-compatible Whisper endpoint for speech-to-text

from sapat.providers import register
from sapat.providers.base import AudioFormat, ProviderConfig
from sapat.providers.openai_compat import OpenAICompatProvider


@register
class LemonfoxProvider(OpenAICompatProvider):
    name = "lemonfox"
    config = ProviderConfig(
        required_env_vars=["LEMONFOX_API_KEY"],
        default_model="whisper-1",
    )
    base_url = "https://api.lemonfox.ai/v1/audio/transcriptions"
    _env_key_for_auth = "LEMONFOX_API_KEY"
