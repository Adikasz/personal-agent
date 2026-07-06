"""Unit tests for `core.config`."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from core.config import Settings, get_settings


class TestSettingsLoading:
    """Environment-driven construction of the `Settings` object."""

    def test_loads_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")
        settings = Settings(_env_file=None)
        assert settings.anthropic_api_key.get_secret_value() == "sk-test-123"

    def test_missing_api_key_raises_validation_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ValidationError):
            Settings(_env_file=None)

    def test_env_variables_are_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("anthropic_api_key", "sk-lower")
        settings = Settings(_env_file=None)
        assert settings.anthropic_api_key.get_secret_value() == "sk-lower"


class TestSecretMasking:
    """The API key must never leak through repr / str."""

    def test_secret_is_masked_in_repr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret-value")
        settings = Settings(_env_file=None)
        assert "sk-secret-value" not in repr(settings)
        assert "sk-secret-value" not in str(settings)


class TestDefaults:
    """Sensible defaults must be applied when env vars are absent."""

    def test_default_model_identifier(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        settings = Settings(_env_file=None)
        assert settings.anthropic_model == "claude-opus-4-7"

    def test_default_max_tokens(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        settings = Settings(_env_file=None)
        assert settings.anthropic_max_tokens == 2048

    def test_default_log_level(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        settings = Settings(_env_file=None)
        assert settings.log_level == "INFO"

    def test_default_max_history_turns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        settings = Settings(_env_file=None)
        assert settings.max_history_turns == 20


class TestValidation:
    """Field-level validators guard against nonsense values."""

    def test_zero_max_tokens_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("ANTHROPIC_MAX_TOKENS", "0")
        with pytest.raises(ValidationError):
            Settings(_env_file=None)

    def test_overly_large_max_tokens_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("ANTHROPIC_MAX_TOKENS", "99999")
        with pytest.raises(ValidationError):
            Settings(_env_file=None)

    def test_zero_history_turns_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("MAX_HISTORY_TURNS", "0")
        with pytest.raises(ValidationError):
            Settings(_env_file=None)

    def test_history_turns_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("MAX_HISTORY_TURNS", "6")
        settings = Settings(_env_file=None)
        assert settings.max_history_turns == 6


class TestResolvedKnowledgeDir:
    """The knowledge directory must always resolve to an absolute path."""

    def test_absolute_path_is_returned_unchanged(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("KNOWLEDGE_DIR", str(tmp_path))
        settings = Settings(_env_file=None)
        assert settings.resolved_knowledge_dir == tmp_path.resolve()

    def test_relative_path_is_resolved_from_project_root(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("KNOWLEDGE_DIR", "knowledge")
        settings = Settings(_env_file=None)
        assert settings.resolved_knowledge_dir.is_absolute()
        assert settings.resolved_knowledge_dir.name == "knowledge"


class TestGetSettings:
    """The `get_settings` accessor must return a cached singleton."""

    def test_returns_same_instance_on_repeated_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        first = get_settings()
        second = get_settings()
        assert first is second
