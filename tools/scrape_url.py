"""Fetch a URL and return its readable text content.

WAT layer: **Tool** — deterministic execution. The tool wraps a single
`httpx.AsyncClient` GET, discards HTML noise (script, style, template
tags) with BeautifulSoup, collapses runs of blank lines, and returns a
strongly-typed pydantic result. Network failures — DNS, TLS, timeouts,
non-2xx responses — are **never** propagated as exceptions; they are
captured on the result's `error` field so the LLM sees the failure and
can pivot on the next iteration without crashing the agent loop.

The scraper never impersonates a browser beyond what public User-Agent
strings advertise; it does not solve CAPTCHAs, bypass paywalls, or
follow login flows. Requests that trigger those flows will return an
error string, which the LLM can surface to the user honestly.
"""

from __future__ import annotations

from typing import Final

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from utils.logger import get_logger

__all__ = [
    "DEFAULT_MAX_CHARS",
    "DEFAULT_TIMEOUT_SECONDS",
    "ScrapeUrlQuery",
    "ScrapeUrlResult",
    "scrape_url",
]

_logger = get_logger(__name__)

DEFAULT_TIMEOUT_SECONDS: Final[float] = 10.0
DEFAULT_MAX_CHARS: Final[int] = 20_000
_STRIPPABLE_TAGS: Final[tuple[str, ...]] = (
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "iframe",
)
_HEADERS: Final[dict[str, str]] = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; PlanSmartAssistant/1.0; "
        "+https://plansmart.live)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


class ScrapeUrlQuery(BaseModel):
    """Structured tool input, enforced by pydantic before the network call."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        frozen=True,
    )

    url: HttpUrl = Field(
        ...,
        description=(
            "Absolute HTTP or HTTPS URL to fetch. Malformed inputs are "
            "refused with a ValidationError."
        ),
    )


class ScrapeUrlResult(BaseModel):
    """Return payload handed back to the Agent layer.

    On success `text` carries the extracted readable content, `error`
    is `None`, and `status_code` reflects the HTTP response. On
    failure `text` is empty, `error` carries a human-readable message,
    and `status_code` is `None` unless a non-2xx response was received.
    """

    model_config = ConfigDict(frozen=True)

    url: str
    text: str
    status_code: int | None = None
    truncated: bool = False
    error: str | None = None


async def scrape_url(
    query: ScrapeUrlQuery,
    *,
    client: httpx.AsyncClient | None = None,
    max_chars: int | None = None,
    timeout_seconds: float | None = None,
) -> ScrapeUrlResult:
    """Fetch `query.url` and return its readable text.

    Args:
        query: Pre-validated tool input.
        client: Test-only override for the `httpx.AsyncClient`. When
            omitted a fresh client is constructed and closed before the
            function returns.
        max_chars: Test-only override for the extract cap.
        timeout_seconds: Test-only override for the HTTP timeout.

    Returns:
        A `ScrapeUrlResult`. Network faults are captured on `error`
        rather than raised, so the agent loop remains stable.
    """
    url_str = str(query.url)
    limit = max_chars if max_chars is not None else DEFAULT_MAX_CHARS
    timeout = timeout_seconds if timeout_seconds is not None else DEFAULT_TIMEOUT_SECONDS

    close_client = client is None
    active_client = client or httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        headers=_HEADERS,
    )
    try:
        try:
            response = await active_client.get(url_str)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            message = (
                f"HTTP {exc.response.status_code} from {url_str}"
            )
            _logger.warning("Scrape rejected: %s", message)
            return ScrapeUrlResult(
                url=url_str,
                text="",
                status_code=exc.response.status_code,
                truncated=False,
                error=message,
            )
        except httpx.HTTPError as exc:
            message = f"{type(exc).__name__}: {exc}"
            _logger.warning("Scrape failed for %s: %s", url_str, message)
            return ScrapeUrlResult(
                url=url_str,
                text="",
                status_code=None,
                truncated=False,
                error=message,
            )
        except Exception as exc:  # noqa: BLE001 — unknown errors must reach the LLM
            message = f"{type(exc).__name__}: {exc}"
            _logger.warning("Scrape failed for %s: %s", url_str, message)
            return ScrapeUrlResult(
                url=url_str,
                text="",
                status_code=None,
                truncated=False,
                error=message,
            )

        text = _extract_text(response.text)
        truncated = len(text) > limit
        if truncated:
            text = text[:limit]
        _logger.info(
            "Scraped %s (%d chars, truncated=%s).",
            url_str,
            len(text),
            truncated,
        )
        return ScrapeUrlResult(
            url=url_str,
            text=text,
            status_code=response.status_code,
            truncated=truncated,
            error=None,
        )
    finally:
        if close_client:
            await active_client.aclose()


def _extract_text(html: str) -> str:
    """Strip HTML noise and return a readable text digest."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(list(_STRIPPABLE_TAGS)):
        tag.decompose()
    raw = soup.get_text(separator="\n", strip=True)
    lines = [line for line in raw.splitlines() if line.strip()]
    return "\n".join(lines)
