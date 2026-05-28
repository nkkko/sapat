# ABOUTME: Package init - loads env vars and exports public API
# ABOUTME: load_dotenv() called here so providers can read env vars at import time

from dotenv import load_dotenv

load_dotenv()

from .__version__ import __version__
from .cli import main

__all__ = ["__version__", "main"]
