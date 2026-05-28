# ABOUTME: Tests for the TranscriptionProvider base class

import os
from unittest.mock import patch

import pytest

from sapat.providers.base import (
    AudioFormat,
    ProviderConfig,
    TranscriptionProvider,
    TranscriptionResult,
)


class MinimalProvider(TranscriptionProvider):
    name = "minimal"

    def transcribe(self, audio_file, model, language="en", prompt=None, temperature=0, **kwargs):
        return TranscriptionResult(text="hello")


class ProviderWithEnv(TranscriptionProvider):
    name = "needs_key"
    config = ProviderConfig(required_env_vars=["TEST_API_KEY"])

    def transcribe(self, audio_file, model, language="en", prompt=None, temperature=0, **kwargs):
        return TranscriptionResult(text="hello")


class ProviderWithPkg(TranscriptionProvider):
    name = "needs_pkg"
    config = ProviderConfig(required_packages=["nonexistent_pkg_xyz"])

    def transcribe(self, audio_file, model, language="en", prompt=None, temperature=0, **kwargs):
        return TranscriptionResult(text="hello")


class TestTranscriptionProvider:
    def test_missing_name_raises(self):
        with pytest.raises(ValueError, match="must set 'name'"):
            class NoName(TranscriptionProvider):
                def transcribe(self, **kw):
                    pass
            NoName()

    def test_minimal_provider_instantiates(self):
        p = MinimalProvider()
        assert p.name == "minimal"

    def test_default_resolve_model(self):
        p = MinimalProvider()
        assert p.resolve_model("whisper") == "whisper"

    def test_default_correct_raises(self):
        p = MinimalProvider()
        with pytest.raises(NotImplementedError):
            p.correct_transcript("some text")

    def test_is_available_no_requirements(self):
        assert MinimalProvider.is_available() is True

    def test_is_available_missing_env_var(self):
        with patch.dict(os.environ, {}, clear=True):
            assert ProviderWithEnv.is_available() is False

    def test_is_available_with_env_var(self):
        with patch.dict(os.environ, {"TEST_API_KEY": "abc"}, clear=False):
            assert ProviderWithEnv.is_available() is True

    def test_is_available_missing_package(self):
        assert ProviderWithPkg.is_available() is False


class TestTranscriptionResult:
    def test_text_only(self):
        r = TranscriptionResult(text="hello world")
        assert r.text == "hello world"
        assert r.language is None
        assert r.segments is None

    def test_full_result(self):
        r = TranscriptionResult(
            text="hello",
            language="en",
            duration=5.0,
            segments=[{"start": 0, "end": 5, "text": "hello"}],
        )
        assert r.language == "en"
        assert r.duration == 5.0
        assert len(r.segments) == 1


class TestProviderConfig:
    def test_defaults(self):
        c = ProviderConfig()
        assert c.required_env_vars == []
        assert c.required_packages == []
        assert c.max_file_size_mb == 25.0
        assert c.preferred_format == AudioFormat.MP3
        assert c.supports_correction is False

    def test_custom_config(self):
        c = ProviderConfig(
            required_env_vars=["KEY"],
            max_file_size_mb=100.0,
            preferred_format=AudioFormat.WAV,
            supports_correction=True,
        )
        assert c.max_file_size_mb == 100.0
        assert c.preferred_format == AudioFormat.WAV
        assert c.supports_correction is True
