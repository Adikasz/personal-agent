"""Perform a DuckDuckGo web search from within the agent loop.

WAT layer: **Tool** — deterministic execution. The tool encapsulates a
single call to the `duckduckgo-search` package and returns a
strongly-typed pydantic result. Network failures are **never**
propagated as exceptions; they are captured on the result's `error`
field so the LLM sees the failure and can pivot on the next iteration
without crashing the agent loop.

The upstream DDGS call is synchronous but is dispatched through
`asyncio.to_thread` so the assistant's asyncio event loop is never
blocked while waiting for search results — an Enterprise-grade concern
that pays off as soon as parallel tools or streaming responses are
introduced.
"""

from __future__ import annotations

import asyncio
from typing import Any, Final

from duckduckgo_search import DDGS
from pydantic import BaseModel, ConfigDict, Field, field_validator

from utils.logger import get_logger

__all__ = [
    "DEFAULT_MAX_RESULTS",
    "SearchHit",
    "WebSearchQuery",
    "WebSearchResult",
    "web_search",
]

_logger = get_logger(__name__)

DEFAULT_MAX_RESULTS: Final[int] = 5
_MAX_QUERY_LENGTH: Final[int] = 200
_ABSOLUTE_MAX_RESULTS: Final[int] = 20


class WebSearchQuery(BaseModel):
    """Structured tool input, enforced by pydantic before the network call."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        frozen=True,
    )

    query: str = Field(
        ...,
        min_length=1,
        max_length=_MAX_QUERY_LENGTH,
        description=(
            "Free-form search query submitted to DuckDuckGo. "
            "Whitespace-only queries are rejected."
        ),
    )
    max_results: int = Field(
        default=DEFAULT_MAX_RESULTS,
        ge=1,
        le=_ABSOLUTE_MAX_RESULTS,
        description="Upper bound on the number of hits to return.",
    )

    @field_validator("query")
    @classmethod
    def _query_is_non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must contain at least one non-whitespace character")
        return value


class SearchHit(BaseModel):
    """A single result row returned by DDGS."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    title: str
    href: str
    body: str


class WebSearchResult(BaseModel):
    """Return payload handed back to the Agent layer.

    On success the tool populates `results`; `error` is `None`. On
    failure the tool populates `error` with a human-readable message
    and leaves `results` empty. The two states are always
    distinguishable by the LLM without inspecting an exception type.
    """

    model_config = ConfigDict(frozen=True)

    query: str
    results: list[SearchHit]
    error: str | None = None


async def web_search(query: WebSearchQuery) -> WebSearchResult:
    """Return DuckDuckGo hits for `query.query`, capped at `query.max_results`.

    The blocking DDGS call is offloaded to a background thread so it
    does not stall the agent's asyncio event loop.

    Args:
        query: Pre-validated tool input. Callers must convert an LLM
            payload via `WebSearchQuery.model_validate(...)` before
            invocation.

    Returns:
        A `WebSearchResult`. Network / rate-limit failures are captured
        on the `error` field rather than raised.
    """
    try:
        raw = await asyncio.to_thread(_run_ddgs, query.query, query.max_results)
    except Exception as exc:  # noqa: BLE001 — network faults must reach the LLM
        message = f"{type(exc).__name__}: {exc}"
        _logger.warning("DDGS failed for %r: %s", query.query, message)
        return WebSearchResult(query=query.query, results=[], error=message)

    hits: list[SearchHit] = []
    for item in raw:
        try:
            hits.append(SearchHit.model_validate(item))
        except Exception as exc:  # noqa: BLE001 — skip malformed row, keep going
            _logger.warning("Skipping malformed DDGS row %r: %s", item, exc)
            continue

    _logger.info(
        "DDGS returned %d hit(s) for %r (requested max %d).",
        len(hits),
        query.query,
        query.max_results,
    )
    return WebSearchResult(query=query.query, results=hits, error=None)


def _run_ddgs(query: str, max_results: int) -> list[dict[str, Any]]:
    """Execute the synchronous DDGS call in a thread-safe fashion.

    Kept as a separate function so tests can patch `DDGS` at the module
    level while `asyncio.to_thread` remains unmocked.
    """
    return list(DDGS().text(query, max_results=max_results))
