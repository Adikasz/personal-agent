"""Centralized logging configuration.

WAT layer: **Tool** — deterministic infrastructure with no LLM calls and
no reasoning. Uses the standard library `logging` module — no `print`
statements are allowed anywhere in the codebase. `configure_logging()`
should be called once from the application entry point; individual
modules obtain their logger via `get_logger(__name__)`.

Handlers installed by this module are tagged with a private sentinel
attribute so that third-party handlers on the root logger (most notably
pytest's ``caplog``) are never removed by our idempotent re-configuration
logic. This lets tests use ``caplog`` without our setup silently stripping
its capture handler.
"""

from __future__ import annotations

import logging
import sys
from typing import Final

_LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
_MANAGED_MARKER: Final[str] = "_plansmart_managed"

_configured: bool = False


def configure_logging(level: str = "INFO") -> None:
    """Initialize the root logger.

    Idempotent: subsequent calls are no-ops until `_configured` is reset
    (test-only). Handlers installed by this function are tagged so that
    a re-configuration removes only our own handlers while leaving any
    third-party handler — such as pytest's ``caplog`` — attached.

    Args:
        level: Textual log level (e.g. ``"DEBUG"``, ``"INFO"``,
            ``"WARNING"``). Unknown values fall back to ``"INFO"``.
    """
    global _configured
    if _configured:
        return

    numeric_level = logging.getLevelName(level.upper())
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))
    _mark_managed(handler)

    root_logger = logging.getLogger()
    root_logger.handlers[:] = [
        existing for existing in root_logger.handlers if not _is_managed(existing)
    ]
    root_logger.addHandler(handler)
    root_logger.setLevel(numeric_level)

    # Silence overly chatty third-party libraries.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, configuring logging on first use if needed."""
    if not _configured:
        configure_logging()
    return logging.getLogger(name)


def _mark_managed(handler: logging.Handler) -> None:
    """Tag `handler` as installed by this module."""
    setattr(handler, _MANAGED_MARKER, True)


def _is_managed(handler: logging.Handler) -> bool:
    """Return True if `handler` carries this module's ownership marker."""
    return bool(getattr(handler, _MANAGED_MARKER, False))
