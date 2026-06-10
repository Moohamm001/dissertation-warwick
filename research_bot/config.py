"""Shared, non-path configuration for research_bot.

NOTE on test isolation: filesystem path constants intentionally live in ``state.py`` and
``discovery.py`` (their own module-level constants), NOT here. Tests must monkeypatch the path
constants in *both* modules to fully redirect the bot away from the real literature brain.
"""
from __future__ import annotations

from pathlib import Path

OPENALEX_BASE = "https://api.openalex.org/works"
# Polite-pool identifier: OpenAlex gives faster, more reliable service to identified callers.
MAILTO = "mooham.00771@gmail.com"

DEFAULT_PER_PAGE = 25
REQUEST_TIMEOUT = 20  # seconds

PACKAGE_DIR = Path(__file__).resolve().parent
SEEDS_PATH = PACKAGE_DIR / "seeds.yaml"
