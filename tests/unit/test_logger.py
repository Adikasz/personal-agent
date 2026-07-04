"""Unit tests for `utils.logger`."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest

from utils import logger as logger_module
from utils.logger import configure_logging, get_logger


@pytest.fixture(autouse=True)
def _reset_logging_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Restore the root logger and the internal guard between tests.

    Only handlers this module installed on a prior test iteration are
    stripped; third-party handlers (notably pytest's ``caplog``) stay
    attached so tests that rely on record capture behave correctly.
    """
    monkeypatch.setattr(logger_module, "_configured", False)
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    root.handlers[:] = [
        handler
        for handler in root.handlers
        if not logger_module._is_managed(handler)
    ]
    try:
        yield
    finally:
        root.handlers = original_handlers
        root.setLevel(original_level)


def _managed_handlers() -> list[logging.Handler]:
    """Return only the handlers this module installed on the root logger."""
    return [h for h in logging.getLogger().handlers if logger_module._is_managed(h)]


class TestConfigureLogging:
    """`configure_logging` must produce a predictable root logger."""

    def test_installs_a_single_stream_handler(self) -> None:
        configure_logging(level="INFO")
        managed = _managed_handlers()
        assert len(managed) == 1
        assert isinstance(managed[0], logging.StreamHandler)

    def test_applies_the_requested_level(self) -> None:
        configure_logging(level="ERROR")
        assert logging.getLogger().level == logging.ERROR

    def test_invalid_level_falls_back_to_info(self) -> None:
        configure_logging(level="NOT_A_REAL_LEVEL")
        assert logging.getLogger().level == logging.INFO

    def test_is_idempotent(self) -> None:
        configure_logging(level="INFO")
        managed_count = len(_managed_handlers())
        configure_logging(level="DEBUG")
        assert len(_managed_handlers()) == managed_count

    def test_silences_chatty_third_party_loggers(self) -> None:
        configure_logging(level="DEBUG")
        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("httpcore").level == logging.WARNING

    def test_formatted_records_contain_level_name_and_message(self) -> None:
        """Verify the formatter output rather than inspecting private state."""
        configure_logging(level="INFO")
        managed = _managed_handlers()
        assert managed, "configure_logging must install at least one handler"
        formatter = managed[0].formatter
        assert formatter is not None

        record = logging.LogRecord(
            name="plansmart.fmt_check",
            level=logging.INFO,
            pathname=__file__,
            lineno=0,
            msg="format probe",
            args=(),
            exc_info=None,
        )
        formatted = formatter.format(record)

        assert "INFO" in formatted
        assert "plansmart.fmt_check" in formatted
        assert "format probe" in formatted

    def test_reconfiguration_does_not_evict_third_party_handlers(self) -> None:
        """Handlers installed by other code (e.g., pytest's ``caplog``) must
        survive our re-configuration path unchanged."""
        foreign = logging.NullHandler()
        logging.getLogger().addHandler(foreign)
        configure_logging(level="INFO")
        assert foreign in logging.getLogger().handlers


class TestGetLogger:
    """`get_logger` must return a fully-usable named logger."""

    def test_returns_a_named_logger(self) -> None:
        log = get_logger("plansmart.test")
        assert isinstance(log, logging.Logger)
        assert log.name == "plansmart.test"

    def test_auto_configures_on_first_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(logger_module, "_configured", False)
        get_logger("plansmart.autoconfig")
        assert logger_module._configured is True

    def test_emitted_records_are_captured(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        configure_logging(level="INFO")
        log = get_logger("plansmart.emit")
        with caplog.at_level(logging.INFO, logger="plansmart.emit"):
            log.info("hello from test")
        assert any("hello from test" in record.message for record in caplog.records)
