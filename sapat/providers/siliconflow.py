# ABOUTME: SiliconFlow transcription provider
# ABOUTME: Uses SiliconFlow's OpenAI-compatible audio transcription endpoint

from sapat.providers import register
from sapat.providers.base import AudioFormat, ProviderConfig
from sapat.providers.openai_compat import OpenAICompatProvider


@register
class SiliconFlowProvider(OpenAICompatProvider):
    name = "siliconflow"
    base_url = "https://api.siliconflow.cn/v1/audio/transcriptions"
    _env_key_for_auth = "SILICONFLOW_API_KEY"
    config = ProviderConfig(
        required_env_vars=["SILICONFLOW_API_KEY"],
        max_file_size_mb=50.0,
        preferred_format=AudioFormat.MP3,
        supports_correction=False,
        default_model="FunAudioLLM/SenseVoiceSmall",
    )

    def _build_data(
        self, model: str, language: str, prompt, temperature: float, **kwargs
    ) -> dict:
        return {"model": model}

    def resolve_model(self, model_alias: str) -> str:
        aliases = {
            "sensevoice": "FunAudioLLM/SenseVoiceSmall",
            "sensevoice-small": "FunAudioLLM/SenseVoiceSmall",
            "teleai": "TeleAI/TeleSpeechASR",
            "telespeech": "TeleAI/TeleSpeechASR",
        }
        return aliases.get(model_alias, model_alias)
