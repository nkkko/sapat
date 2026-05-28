# ABOUTME: Abstract base class for transcription providers
# ABOUTME: Defines the interface all providers must implement

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class AudioFormat(Enum):
    MP3 = "mp3"
    WAV = "wav"
    FLAC = "flac"
    OGG = "ogg"


@dataclass
class TranscriptionResult:
    text: str
    language: Optional[str] = None
    duration: Optional[float] = None
    segments: Optional[List[Dict[str, Any]]] = None
    raw_response: Optional[Any] = None


@dataclass
class ProviderConfig:
    required_env_vars: List[str] = field(default_factory=list)
    required_packages: List[str] = field(default_factory=list)
    extras_key: Optional[str] = None
    max_file_size_mb: float = 25.0
    preferred_format: AudioFormat = AudioFormat.MP3
    supports_correction: bool = False
    default_model: str = ""


class TranscriptionProvider(ABC):
    name: str = ""
    config: ProviderConfig = ProviderConfig()

    def __init__(self):
        if not self.name:
            raise ValueError(f"{self.__class__.__name__} must set 'name'")

    @abstractmethod
    def transcribe(
        self,
        audio_file: str,
        model: str,
        language: str = "en",
        prompt: Optional[str] = None,
        temperature: float = 0,
        **kwargs,
    ) -> TranscriptionResult:
        ...

    def correct_transcript(self, text: str, temperature: float = 0) -> str:
        raise NotImplementedError(
            f"Provider '{self.name}' does not implement transcript correction"
        )

    def resolve_model(self, model_alias: str) -> str:
        return model_alias

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
