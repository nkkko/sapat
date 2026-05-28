# ABOUTME: Shared test fixtures for sapat test suite

import os
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def isolate_env():
    """Ensure each test gets a clean environment."""
    original = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original)
