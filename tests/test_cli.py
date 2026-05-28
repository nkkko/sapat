# ABOUTME: Tests for the Click CLI integration

import os
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from sapat.providers.base import (
    ProviderConfig,
    TranscriptionProvider,
    TranscriptionResult,
)


class MockProvider(TranscriptionProvider):
    name = "mock"
    config = ProviderConfig(default_model="mock-model")

    def transcribe(self, audio_file, model, language="en", prompt=None, temperature=0, **kwargs):
        return TranscriptionResult(text="mocked transcription")


@pytest.fixture(autouse=True)
def mock_registry(monkeypatch):
    """Replace the registry with a single mock provider."""
    import sapat.providers as reg
    reg._registry.clear()
    reg._discovered = True
    reg._registry["mock"] = MockProvider

    monkeypatch.setattr("sapat.providers._discover_providers", lambda: None)
    monkeypatch.setattr("sapat.providers.get_available_providers", lambda: {"mock": MockProvider})

    yield

    reg._registry.clear()
    reg._discovered = False


class TestCLI:
    def test_help_shows_provider_option(self):
        from sapat.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "--provider" in result.output

    def test_version_flag(self):
        from sapat.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert "0.3.0" in result.output

    def test_no_providers_available_error(self, monkeypatch):
        from sapat.cli import main
        monkeypatch.setattr("sapat.providers.get_available_providers", lambda: {})
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        # Should not crash at help time, but invocation would fail
        assert result.exit_code == 0
