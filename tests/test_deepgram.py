from unittest.mock import Mock

import pytest
import requests

from sapat.providers.deepgram import DeepgramProvider


@pytest.fixture
def audio(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-only-key")
    path = tmp_path / "speech.mp3"
    path.write_bytes(b"test audio bytes")
    return path


def payload(text="Hello, world."):
    return {
        "metadata": {"duration": 1.25},
        "results": {
            "channels": [
                {
                    "detected_language": "en",
                    "alternatives": [{"transcript": text}],
                }
            ]
        },
    }


def test_binary_upload_and_response(audio, monkeypatch):
    captured = {}

    def post(url, **kwargs):
        captured.update(url=url, **kwargs)
        assert kwargs["data"].read() == audio.read_bytes()
        return Mock(status_code=200, json=lambda: payload())

    monkeypatch.setattr(requests, "post", post)
    result = DeepgramProvider().transcribe(str(audio), "nova-3", language="en-GB")
    assert captured["url"] == "https://api.deepgram.com/v1/listen"
    assert captured["headers"] == {
        "Authorization": "Token test-only-key",
        "Content-Type": "audio/mpeg",
    }
    assert captured["params"] == {
        "model": "nova-3",
        "language": "en-GB",
        "smart_format": "true",
    }
    assert captured["timeout"] == (10, 300)
    assert captured["data"].closed
    assert result.text == "Hello, world."
    assert result.language == "en"
    assert result.duration == 1.25


def test_auto_language_and_empty_speech(audio, monkeypatch, caplog):
    post = Mock(return_value=Mock(status_code=200, json=lambda: payload("")))
    monkeypatch.setattr(requests, "post", post)
    result = DeepgramProvider().transcribe(
        str(audio), "nova-3", language="", prompt="hint", temperature=0.5
    )
    assert result.text == ""
    assert post.call_args.kwargs["params"] == {
        "model": "nova-3",
        "smart_format": "true",
        "detect_language": "true",
    }
    assert "ignored" in caplog.text


@pytest.mark.parametrize("status", [400, 401, 429, 500])
def test_http_errors_do_not_echo_body(audio, monkeypatch, status):
    monkeypatch.setattr(
        requests,
        "post",
        Mock(return_value=Mock(status_code=status, text="private audio content")),
    )
    with pytest.raises(RuntimeError, match=f"HTTP {status}") as exc:
        DeepgramProvider().transcribe(str(audio), "nova-3")
    assert "private" not in str(exc.value)


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"results": {"channels": []}},
        {"results": {"channels": [{"alternatives": []}]}},
        payload(None),
    ],
)
def test_malformed_response_is_not_empty_success(audio, monkeypatch, data):
    monkeypatch.setattr(
        requests, "post", Mock(return_value=Mock(status_code=200, json=lambda: data))
    )
    with pytest.raises(RuntimeError, match="invalid transcription response"):
        DeepgramProvider().transcribe(str(audio), "nova-3")


def test_missing_key_fails_before_network(audio, monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_KEY")
    post = Mock()
    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(ValueError, match="DEEPGRAM_API_KEY"):
        DeepgramProvider().transcribe(str(audio), "nova-3")
    post.assert_not_called()


def test_timeout_closes_audio(audio, monkeypatch):
    opened = []

    def post(url, **kwargs):
        opened.append(kwargs["data"])
        raise requests.Timeout("timed out")

    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(requests.Timeout):
        DeepgramProvider().transcribe(str(audio), "nova-3")
    assert opened[0].closed


def test_invalid_json(audio, monkeypatch):
    response = Mock(status_code=200)
    response.json.side_effect = ValueError("not JSON")
    monkeypatch.setattr(requests, "post", Mock(return_value=response))
    with pytest.raises(RuntimeError, match="invalid transcription response"):
        DeepgramProvider().transcribe(str(audio), "nova-3")


def test_provider_discovery_in_fresh_process(audio):
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from sapat.providers import get_provider; "
            "p = get_provider('deepgram'); assert p is not None; "
            "assert p.config.default_model == 'nova-3'",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
