"""Unit tests for `tools.scrape_url`.

`httpx.AsyncClient` is injected via the tool's `client` kwarg so no
network I/O occurs. Every code path — pydantic validation, happy path
with HTML sanitization, HTTP error responses, transport failures, and
truncation — is exercised deterministically.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from pydantic import ValidationError

from tools.scrape_url import (
    DEFAULT_MAX_CHARS,
    DEFAULT_TIMEOUT_SECONDS,
    ScrapeUrlQuery,
    ScrapeUrlResult,
    scrape_url,
)


def _fake_client(
    *,
    html: str | None = None,
    status_code: int = 200,
    raise_on_get: Exception | None = None,
    raise_for_status: Exception | None = None,
) -> AsyncMock:
    """Build a minimal AsyncMock that satisfies the httpx.AsyncClient
    contract used by `scrape_url`."""
    response = MagicMock()
    response.text = html or ""
    response.status_code = status_code
    if raise_for_status is not None:
        response.raise_for_status = MagicMock(side_effect=raise_for_status)
    else:
        response.raise_for_status = MagicMock()

    client = AsyncMock()
    if raise_on_get is not None:
        client.get = AsyncMock(side_effect=raise_on_get)
    else:
        client.get = AsyncMock(return_value=response)
    client.aclose = AsyncMock()
    return client


def _q(url: str) -> ScrapeUrlQuery:
    """Build a query from a raw URL string via the canonical validator.

    ``ScrapeUrlQuery.url`` is typed ``HttpUrl``, so the generated
    constructor signature expects an already-parsed URL. Routing through
    ``model_validate`` performs the exact string->HttpUrl coercion the
    agent relies on at runtime while keeping the call type-correct.
    """
    return ScrapeUrlQuery.model_validate({"url": url})


class TestQueryValidation:
    """`ScrapeUrlQuery` enforces the LLM-facing input contract."""

    def test_valid_https_url_round_trips(self) -> None:
        query = _q("https://example.com/page")
        assert str(query.url).startswith("https://example.com/page")

    def test_valid_http_url_round_trips(self) -> None:
        query = _q("http://example.com")
        assert str(query.url).startswith("http://example.com")

    def test_missing_scheme_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _q("example.com/no-scheme")

    def test_ftp_scheme_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _q("ftp://example.com/file")

    def test_empty_url_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _q("")

    def test_extra_fields_are_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            ScrapeUrlQuery.model_validate({"url": "https://example.com", "nonsense": "x"})

    def test_query_is_immutable(self) -> None:
        query = _q("https://example.com")
        with pytest.raises(ValidationError):
            query.url = "https://tampered.com"  # type: ignore[assignment]


class TestHappyPath:
    """Well-formed HTML must yield sanitized readable text."""

    async def test_returns_extracted_text_on_success(self) -> None:
        html = (
            "<html><head><title>Title</title>"
            "<script>alert('xss')</script>"
            "<style>body { color: red; }</style></head>"
            "<body><h1>Heading</h1><p>Paragraph one.</p>"
            "<p>Paragraph two.</p></body></html>"
        )
        client = _fake_client(html=html, status_code=200)

        result = await scrape_url(
            _q("https://example.com/x"),
            client=client,
        )

        assert isinstance(result, ScrapeUrlResult)
        assert result.error is None
        assert result.status_code == 200
        assert result.truncated is False
        assert "alert('xss')" not in result.text
        assert "color: red" not in result.text
        assert "Heading" in result.text
        assert "Paragraph one." in result.text
        assert "Paragraph two." in result.text

    async def test_strips_all_declared_noisy_tags(self) -> None:
        html = (
            "<html><body>"
            "<noscript>fallback</noscript>"
            "<template>tmpl</template>"
            "<svg>svg-content</svg>"
            "<iframe>iframe-content</iframe>"
            "<p>Kept.</p>"
            "</body></html>"
        )
        client = _fake_client(html=html)

        result = await scrape_url(
            _q("https://example.com"),
            client=client,
        )

        for stripped in ("fallback", "tmpl", "svg-content", "iframe-content"):
            assert stripped not in result.text
        assert "Kept." in result.text

    async def test_collapses_blank_lines(self) -> None:
        html = "<html><body><p>a</p><p></p><p>b</p></body></html>"
        client = _fake_client(html=html)
        result = await scrape_url(
            _q("https://example.com"),
            client=client,
        )
        assert result.text == "a\nb"

    async def test_default_client_is_closed_when_no_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the tool constructs its own client it must aclose it."""
        # Patch httpx.AsyncClient so we can spy on the constructor and the
        # aclose call without exercising the network. monkeypatch restores
        # the original attribute automatically at teardown.
        html = "<html><body>ok</body></html>"
        constructed_clients: list[AsyncMock] = []

        def _factory(**_kwargs: object) -> AsyncMock:
            client = _fake_client(html=html)
            constructed_clients.append(client)
            return client

        monkeypatch.setattr(httpx, "AsyncClient", _factory)
        result = await scrape_url(_q("https://example.com/page"))

        assert result.text == "ok"
        assert len(constructed_clients) == 1
        constructed_clients[0].aclose.assert_awaited_once()


class TestHttpFailureResponses:
    """A non-2xx response must be captured on `error`, never raised."""

    async def test_http_status_error_captures_status_code(self) -> None:
        response = MagicMock()
        response.status_code = 404
        error = httpx.HTTPStatusError("not found", request=MagicMock(), response=response)
        client = _fake_client(html="", raise_for_status=error)
        # Also make `response.status_code` accessible via the mocked
        # response returned from `.get`; simulate this by wiring the
        # error's response directly.
        client.get = AsyncMock(
            return_value=MagicMock(
                text="",
                status_code=404,
                raise_for_status=MagicMock(side_effect=error),
            )
        )

        result = await scrape_url(
            _q("https://example.com/missing"),
            client=client,
        )

        assert result.text == ""
        assert result.status_code == 404
        assert result.error is not None
        assert "HTTP 404" in result.error


class TestTransportFailures:
    """DNS, timeout, and TLS faults must all fall back to `error`."""

    async def test_timeout_is_captured_as_error(self) -> None:
        client = _fake_client(raise_on_get=httpx.TimeoutException("read timed out"))
        result = await scrape_url(
            _q("https://slow.example.com"),
            client=client,
        )
        assert result.text == ""
        assert result.status_code is None
        assert result.error is not None
        assert "TimeoutException" in result.error

    async def test_connection_error_is_captured_as_error(self) -> None:
        client = _fake_client(raise_on_get=httpx.ConnectError("dns failure"))
        result = await scrape_url(
            _q("https://bad.example.com"),
            client=client,
        )
        assert result.error is not None
        assert "ConnectError" in result.error

    async def test_unexpected_exception_is_captured_as_error(self) -> None:
        client = _fake_client(raise_on_get=RuntimeError("kernel panic"))
        result = await scrape_url(
            _q("https://example.com"),
            client=client,
        )
        assert result.error is not None
        assert "RuntimeError" in result.error
        assert "kernel panic" in result.error


class TestTruncation:
    """Oversized documents must be capped, and `truncated` set to True."""

    async def test_truncation_is_applied_beyond_the_limit(self) -> None:
        payload = "abcdefghij" * 100  # 1000 characters when extracted
        html = f"<html><body><p>{payload}</p></body></html>"
        client = _fake_client(html=html)

        result = await scrape_url(
            _q("https://example.com"),
            client=client,
            max_chars=50,
        )
        assert result.truncated is True
        assert len(result.text) == 50

    def test_default_max_chars_and_timeout_are_reasonable(self) -> None:
        assert 5_000 <= DEFAULT_MAX_CHARS <= 200_000
        assert 1.0 <= DEFAULT_TIMEOUT_SECONDS <= 60.0
