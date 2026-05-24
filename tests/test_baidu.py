import os
import tempfile
import unittest
from unittest.mock import patch

from sapat.transcription.baidu import BaiduTranscription


class FakeResponse:
    def __init__(self, payload, status_code=200, text=""):
        self.payload = payload
        self.status_code = status_code
        self.text = text

    def json(self):
        return self.payload


class BaiduTranscriptionTest(unittest.TestCase):
    def make_audio_file(self):
        audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        audio.write(b"fake audio bytes")
        audio.close()
        self.addCleanup(os.unlink, audio.name)
        return audio.name

    @patch.dict(os.environ, {"BAIDU_API_KEY": "key", "BAIDU_SECRET_KEY": "secret"}, clear=True)
    @patch("sapat.transcription.baidu.requests.post")
    @patch("sapat.transcription.baidu.requests.get")
    def test_transcribes_with_baidu_payload(self, mock_get, mock_post):
        mock_get.return_value = FakeResponse({"access_token": "token"})
        mock_post.return_value = FakeResponse({"err_no": 0, "result": ["hello world"]})

        transcriber = BaiduTranscription(temperature=0.3)
        result = transcriber.transcribe_audio(self.make_audio_file(), language="en")

        self.assertEqual(result, {"text": "hello world"})
        mock_get.assert_called_once()
        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(payload["token"], "token")
        self.assertEqual(payload["format"], "mp3")
        self.assertEqual(payload["rate"], 16000)
        self.assertEqual(payload["dev_pid"], 1737)
        self.assertEqual(payload["len"], len(b"fake audio bytes"))
        self.assertTrue(payload["speech"])

    @patch.dict(os.environ, {"BAIDU_API_KEY": "key", "BAIDU_SECRET_KEY": "secret"}, clear=True)
    def test_maps_chinese_language_to_mandarin_model(self):
        transcriber = BaiduTranscription(temperature=0.3)

        self.assertEqual(transcriber._resolve_dev_pid("zh-CN"), 1537)

    @patch.dict(os.environ, {}, clear=True)
    def test_requires_credentials(self):
        transcriber = BaiduTranscription(temperature=0.3)

        with self.assertRaisesRegex(ValueError, "BAIDU_API_KEY"):
            transcriber._get_access_token()

    @patch.dict(os.environ, {"BAIDU_API_KEY": "key", "BAIDU_SECRET_KEY": "secret"}, clear=True)
    @patch("sapat.transcription.baidu.requests.post")
    @patch("sapat.transcription.baidu.requests.get")
    def test_raises_on_baidu_error(self, mock_get, mock_post):
        mock_get.return_value = FakeResponse({"access_token": "token"})
        mock_post.return_value = FakeResponse({"err_no": 3301, "err_msg": "audio quality error"})

        transcriber = BaiduTranscription(temperature=0.3)

        with self.assertRaisesRegex(Exception, "audio quality error"):
            transcriber.transcribe_audio(self.make_audio_file(), language="en")


if __name__ == "__main__":
    unittest.main()
