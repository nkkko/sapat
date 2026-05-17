from unittest.mock import MagicMock, patch


def test_default_endpoint_whisper_v3_turbo(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw_test_key")
    monkeypatch.setenv("FIREWORKS_MODEL", "whisper-v3-turbo")
    monkeypatch.delenv("FIREWORKS_API_ENDPOINT", raising=False)

    from importlib import reload
    import sapat.transcription.fireworks as fireworks

    reload(fireworks)
    svc = fireworks.FireworksTranscription(temperature=0.0)

    assert (
        svc.endpoint
        == "https://audio-turbo.api.fireworks.ai/v1/audio/transcriptions"
    )


def test_default_endpoint_non_turbo(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw_test_key")
    monkeypatch.setenv("FIREWORKS_MODEL", "whisper-v3")
    monkeypatch.delenv("FIREWORKS_API_ENDPOINT", raising=False)

    from importlib import reload
    import sapat.transcription.fireworks as fireworks

    reload(fireworks)
    svc = fireworks.FireworksTranscription(temperature=0.0)

    assert svc.endpoint == "https://audio-prod.api.fireworks.ai/v1/audio/transcriptions"


def test_transcribe_audio_json_success(tmp_path, monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw_test_key")
    monkeypatch.setenv("FIREWORKS_MODEL", "whisper-v3")
    monkeypatch.setenv(
        "FIREWORKS_API_ENDPOINT",
        "https://audio-prod.api.fireworks.ai/v1/audio/transcriptions",
    )

    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"fake-mp3-bytes-for-test")

    from importlib import reload
    import sapat.transcription.fireworks as fireworks

    reload(fireworks)

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {"text": "hello"}

    with patch.object(fireworks.requests, "post", return_value=fake_resp) as post:
        svc = fireworks.FireworksTranscription(temperature=0.2)
        out = svc.transcribe_audio(str(audio))

    assert out == {"text": "hello"}
    args, kwargs = post.call_args
    assert kwargs["headers"]["Authorization"] == "fw_test_key"
    payload = kwargs["data"]
    assert payload["response_format"] == "json"


def test_transcribe_raises_on_missing_key(monkeypatch, tmp_path):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    monkeypatch.setenv("FIREWORKS_MODEL", "whisper-v3")
    monkeypatch.delenv("FIREWORKS_API_ENDPOINT", raising=False)

    from importlib import reload
    import sapat.transcription.fireworks as fireworks

    reload(fireworks)

    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"x")

    svc = fireworks.FireworksTranscription(temperature=0.0)
    try:
        svc.transcribe_audio(str(audio))
    except ValueError as e:
        assert "FIREWORKS_API_KEY" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_transcribe_rejects_bad_extension(monkeypatch, tmp_path):
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw_test_key")
    monkeypatch.setenv("FIREWORKS_MODEL", "whisper-v3")
    monkeypatch.delenv("FIREWORKS_API_ENDPOINT", raising=False)

    from importlib import reload
    import sapat.transcription.fireworks as fireworks

    reload(fireworks)

    bad = tmp_path / "audio.m4a"
    bad.write_bytes(b"x")

    svc = fireworks.FireworksTranscription(temperature=0.0)

    try:
        svc.transcribe_audio(str(bad))
    except ValueError as e:
        assert "Unsupported audio file format" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_transcribe_audio_raises_on_http_error(tmp_path, monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw_test_key")
    monkeypatch.setenv("FIREWORKS_MODEL", "whisper-v3")
    monkeypatch.delenv("FIREWORKS_API_ENDPOINT", raising=False)

    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"x")

    from importlib import reload
    import sapat.transcription.fireworks as fireworks

    reload(fireworks)

    fake_resp = MagicMock()
    fake_resp.status_code = 400
    fake_resp.text = "bad request body"

    with patch.object(fireworks.requests, "post", return_value=fake_resp):
        svc = fireworks.FireworksTranscription(temperature=0.0)

        try:
            svc.transcribe_audio(str(audio))
        except Exception as e:
            assert "Transcription failed" in str(e)
            assert "bad request body" in str(e)
        else:
            raise AssertionError("expected Exception")


def test_corrected_transcript_requires_chat_model(monkeypatch, tmp_path):
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw_test_key")
    monkeypatch.setenv("FIREWORKS_MODEL", "whisper-v3")
    monkeypatch.delenv("FIREWORKS_API_ENDPOINT", raising=False)
    monkeypatch.delenv("FIREWORKS_MODEL_NAME_CHAT", raising=False)

    from importlib import reload
    import sapat.transcription.fireworks as fireworks

    reload(fireworks)

    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"x")

    svc = fireworks.FireworksTranscription(temperature=0.0)

    try:
        svc.generate_corrected_transcript(str(audio), 0.5, "prompt")
    except ValueError as e:
        assert "FIREWORKS_MODEL_NAME_CHAT" in str(e)
    else:
        raise AssertionError("expected ValueError")

