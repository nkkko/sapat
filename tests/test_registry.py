# ABOUTME: Tests for the provider registry and auto-discovery

import os
from unittest.mock import patch

import pytest

from sapat.providers import (
    _discover_providers,
    _registry,
    get_available_providers,
    get_provider,
    get_provider_choices,
    register,
)
from sapat.providers.base import (
    ProviderConfig,
    TranscriptionProvider,
    TranscriptionResult,
)


class FakeProvider(TranscriptionProvider):
    name = "fake_test"
    config = ProviderConfig(required_env_vars=[])

    def transcribe(self, audio_file, model, language="en", prompt=None, temperature=0, **kwargs):
        return TranscriptionResult(text="fake")


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the registry before each test."""
    import sapat.providers as reg
    reg._registry.clear()
    reg._discovered = False
    yield
    reg._registry.clear()
    reg._discovered = False


class TestRegister:
    def test_register_available_provider(self):
        register(FakeProvider)
        assert "fake_test" in _registry

    def test_register_unavailable_provider_skipped(self):
        class Unavailable(TranscriptionProvider):
            name = "unavail"
            config = ProviderConfig(required_env_vars=["MISSING_KEY"])

            def transcribe(self, **kw):
                pass

        with patch.dict(os.environ, {}, clear=True):
            register(Unavailable)
        assert "unavail" not in _registry

    def test_duplicate_name_ignored(self):
        register(FakeProvider)
        register(FakeProvider)  # second registration is ignored
        assert _registry["fake_test"] is FakeProvider


class TestGetProvider:
    def test_get_existing(self):
        register(FakeProvider)
        provider = get_provider("fake_test")
        assert isinstance(provider, FakeProvider)

    def test_get_nonexistent_returns_none(self):
        assert get_provider("does_not_exist") is None


class TestGetProviderChoices:
    def test_empty_when_none_registered(self):
        assert get_provider_choices() == []

    def test_returns_sorted_names(self):
        register(FakeProvider)

        class ZProvider(TranscriptionProvider):
            name = "z_last"
            config = ProviderConfig()

            def transcribe(self, **kw):
                pass

        register(ZProvider)
        assert get_provider_choices() == ["fake_test", "z_last"]


class TestAutoDiscovery:
    def test_discovers_azure_when_env_set(self):
        with patch.dict(os.environ, {
            "AZURE_OPENAI_API_KEY": "test",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
            "AZURE_OPENAI_STT_API_VERSION": "2024-02-01",
        }):
            from sapat.providers.azure import AzureProvider
            register(AzureProvider)
            assert "azure" in _registry

    def test_discovers_groq_when_env_set(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "test"}):
            from sapat.providers.groq import GroqProvider
            register(GroqProvider)
            assert "groq" in _registry

    def test_no_discovery_without_env(self):
        with patch.dict(os.environ, {}, clear=True):
            _discover_providers()
            # Should not crash, just skip unavailable providers
            assert isinstance(_registry, dict)
