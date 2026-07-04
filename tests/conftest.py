"""Shared fixtures for the PlanSmart test suite.

Every test starts from a clean state: the cached `Settings` instance is
cleared, any leftover environment variables that could leak from `.env`
into the process are removed, and sensible dummy values are injected
for the three required API keys. Tests that specifically want to
exercise the missing-key path can override the defaults with
`monkeypatch.delenv(...)`.
"""

from __future__ import annotations

import pytest

from core.config import get_settings

_ENV_VARS_UNDER_TEST: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_MAX_TOKENS",
    "MAX_HISTORY_TURNS",
    "MAX_TOOL_ITERATIONS",
    "LOG_LEVEL",
    "KNOWLEDGE_DIR",
    "OPENAI_API_KEY",
    "OPENAI_EMBEDDING_MODEL",
    "PINECONE_API_KEY",
    "PINECONE_INDEX_NAME",
)

_INJECTED_DEFAULTS: dict[str, str] = {
    # Dummies for the required secret keys so tests unrelated to the
    # config layer can construct `Settings` without extra ceremony.
    "OPENAI_API_KEY": "sk-test-openai",
    "PINECONE_API_KEY": "pc-test-pinecone",
}


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guarantee a fresh `Settings` cache and a clean environment per test."""
    for var in _ENV_VARS_UNDER_TEST:
        monkeypatch.delenv(var, raising=False)
    for var, value in _INJECTED_DEFAULTS.items():
        monkeypatch.setenv(var, value)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
