# ABOUTME: Shared mixin for async polling transcription providers
# ABOUTME: Implements the upload -> poll -> fetch result pattern

import time
from abc import abstractmethod
from typing import Optional

from sapat.providers.base import (
    TranscriptionProvider,
    TranscriptionResult,
)


class AsyncPollProvider(TranscriptionProvider):
    """
    Base for providers using an async upload -> poll -> fetch pattern.

    Subclasses implement: _upload(), _poll(), _fetch_result()
    """

    poll_interval: float = 5.0
    max_poll_time: float = 600.0

    @abstractmethod
    def _upload(self, audio_file: str, model: str, language: str, **kwargs) -> str:
        """Upload audio, return job/transaction ID."""
        ...

    @abstractmethod
    def _poll(self, job_id: str) -> str:
        """Check status. Return 'completed', 'failed', or other."""
        ...

    @abstractmethod
    def _fetch_result(self, job_id: str) -> TranscriptionResult:
        """Fetch the completed transcription."""
        ...

    def transcribe(
        self,
        audio_file: str,
        model: str,
        language: str = "en",
        prompt: Optional[str] = None,
        temperature: float = 0,
        **kwargs,
    ) -> TranscriptionResult:
        job_id = self._upload(audio_file, model, language, **kwargs)

        elapsed = 0.0
        while elapsed < self.max_poll_time:
            status = self._poll(job_id)
            if status == "completed":
                return self._fetch_result(job_id)
            if status == "failed":
                raise RuntimeError(f"{self.name} job {job_id} failed")
            time.sleep(self.poll_interval)
            elapsed += self.poll_interval

        raise TimeoutError(f"{self.name} job {job_id} timed out after {self.max_poll_time}s")
