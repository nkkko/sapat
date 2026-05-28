# ABOUTME: Provider registry with auto-discovery
# ABOUTME: Walks sapat/providers/ and registers available providers

import importlib
import logging
import pkgutil
from typing import Dict, List, Optional, Type

from sapat.providers.base import TranscriptionProvider

logger = logging.getLogger(__name__)

_registry: Dict[str, Type[TranscriptionProvider]] = {}
_discovered: bool = False


def _discover_providers():
    global _discovered
    if _discovered:
        return

    _SKIP_MODULES = {"base", "openai_compat", "async_poll"}

    for importer, module_name, is_pkg in pkgutil.iter_modules(__path__):
        if module_name.startswith("_") or module_name in _SKIP_MODULES:
            continue
        try:
            importlib.import_module(f"sapat.providers.{module_name}")
        except Exception as e:
            logger.debug("Skipping provider module %s: %s", module_name, e)

    _discovered = True


def register(cls: Type[TranscriptionProvider]) -> Type[TranscriptionProvider]:
    if not issubclass(cls, TranscriptionProvider):
        raise TypeError(f"{cls} is not a TranscriptionProvider subclass")
    if not cls.is_available():
        logger.debug("Provider %s not available (missing deps/env)", cls.name)
        return cls
    if cls.name in _registry:
        logger.warning(
            "Duplicate provider name %s from %s (already registered by %s)",
            cls.name, cls, _registry[cls.name],
        )
        return cls
    _registry[cls.name] = cls
    return cls


def get_provider(name: str) -> Optional[TranscriptionProvider]:
    _discover_providers()
    cls = _registry.get(name)
    if cls is None:
        return None
    return cls()


def get_available_providers() -> Dict[str, Type[TranscriptionProvider]]:
    _discover_providers()
    return dict(_registry)


def get_provider_choices() -> List[str]:
    _discover_providers()
    return sorted(_registry.keys())
