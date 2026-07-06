"""Unit tests for `tools.web_search`.

The `duckduckgo-search` package is mocked at the class level so no
network I/O occurs. Every code path — pydantic validation, happy path,
rate-limit / network failure, malformed rows — is exercised
deterministically.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from tools.web_search import (
    DEFAULT_MAX_RESULTS,
    SearchHit,
    WebSearchQuery,
    WebSearchResult,
    web_search,
)


class TestQueryValidation:
    """`WebSearchQuery` enforces a strict, minimal input contract."""

    def test_valid_query_round_trips(self) -> None:
        query = WebSearchQuery(query="career architect", max_results=3)
        assert query.query == "career architect"
        assert query.max_results == 3

    def test_default_max_results_is_five(self) -> None:
        query = WebSearchQuery(query="anything")
        assert query.max_results == DEFAULT_MAX_RESULTS == 5

    def test_empty_query_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            WebSearchQuery(query="")

    def test_whitespace_only_query_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            WebSearchQuery(query="   ")

    def test_overly_long_query_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            WebSearchQuery(query="x" * 201)

    def test_max_results_lower_bound(self) -> None:
        with pytest.raises(ValidationError):
            WebSearchQuery(query="x", max_results=0)

    def test_max_results_upper_bound(self) -> None:
        with pytest.raises(ValidationError):
            WebSearchQuery(query="x", max_results=21)

    def test_extra_fields_are_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            WebSearchQuery(query="x", nonsense="drop me")  # type: ignore[call-arg]

    def test_query_is_immutable(self) -> None:
        query = WebSearchQuery(query="hello")
        with pytest.raises(ValidationError):
            query.query = "changed"  # type: ignore[misc]


class TestSuccessfulSearch:
    """Well-formed DDGS output must yield a populated result with no error."""

    async def test_returns_hits_when_ddgs_yields_rows(self) -> None:
        rows = [
            {
                "title": "How to Become a Career Architect",
                "href": "https://example.com/architect",
                "body": "A guide to reshaping your career path.",
            },
            {
                "title": "Career Coaching 101",
                "href": "https://example.com/coaching",
                "body": "Fundamentals of professional coaching.",
            },
        ]
        with patch("tools.web_search.DDGS") as ddgs_cls:
            ddgs_cls.return_value.text.return_value = rows
            result = await web_search(WebSearchQuery(query="career architect"))

        assert isinstance(result, WebSearchResult)
        assert result.query == "career architect"
        assert result.error is None
        assert len(result.results) == 2
        assert result.results[0] == SearchHit(
            title="How to Become a Career Architect",
            href="https://example.com/architect",
            body="A guide to reshaping your career path.",
        )

    async def test_max_results_is_forwarded_to_ddgs(self) -> None:
        with patch("tools.web_search.DDGS") as ddgs_cls:
            ddgs_cls.return_value.text.return_value = []
            await web_search(WebSearchQuery(query="x", max_results=7))

        text_mock = ddgs_cls.return_value.text
        text_mock.assert_called_once_with("x", max_results=7)

    async def test_empty_row_list_yields_empty_results_without_error(self) -> None:
        with patch("tools.web_search.DDGS") as ddgs_cls:
            ddgs_cls.return_value.text.return_value = []
            result = await web_search(WebSearchQuery(query="nothing"))

        assert result.results == []
        assert result.error is None


class TestNetworkFailures:
    """Every raised exception must be captured on `error`, never propagated."""

    async def test_rate_limit_error_is_captured_on_error_field(self) -> None:
        with patch("tools.web_search.DDGS") as ddgs_cls:
            ddgs_cls.return_value.text.side_effect = RuntimeError(
                "DuckDuckGoSearchException: 202 Ratelimit"
            )
            result = await web_search(WebSearchQuery(query="x"))

        assert result.results == []
        assert result.error is not None
        assert "Ratelimit" in result.error
        assert "RuntimeError" in result.error

    async def test_network_error_is_captured_on_error_field(self) -> None:
        with patch("tools.web_search.DDGS") as ddgs_cls:
            ddgs_cls.return_value.text.side_effect = ConnectionError("dns failure")
            result = await web_search(WebSearchQuery(query="x"))

        assert result.results == []
        assert result.error is not None
        assert "ConnectionError" in result.error
        assert "dns failure" in result.error


class TestMalformedRowResilience:
    """Rows missing the DDGS contract are skipped, not fatal."""

    async def test_malformed_rows_are_skipped_and_others_kept(self) -> None:
        rows = [
            {"title": "good", "href": "https://example.com/a", "body": "one"},
            {"malformed": "yes"},  # missing title/href/body — must be skipped
            {"title": "also good", "href": "https://example.com/b", "body": "two"},
        ]
        with patch("tools.web_search.DDGS") as ddgs_cls:
            ddgs_cls.return_value.text.return_value = rows
            result = await web_search(WebSearchQuery(query="x"))

        assert result.error is None
        assert len(result.results) == 2
        assert [hit.title for hit in result.results] == ["good", "also good"]

    async def test_hit_model_ignores_unknown_ddgs_fields(self) -> None:
        rows = [
            {
                "title": "t",
                "href": "https://example.com",
                "body": "b",
                "future_field": "ignored",
            }
        ]
        with patch("tools.web_search.DDGS") as ddgs_cls:
            ddgs_cls.return_value.text.return_value = rows
            result = await web_search(WebSearchQuery(query="x"))

        assert len(result.results) == 1
        assert result.results[0].title == "t"


def _dummy_result_shape() -> WebSearchResult:
    """Sanity-check that WebSearchResult is frozen for LLM-safe reuse."""
    return WebSearchResult(query="x", results=[], error=None)


class TestResultShape:
    """The pydantic return type is frozen and constructor-strict."""

    def test_result_is_frozen(self) -> None:
        result = _dummy_result_shape()
        with pytest.raises(ValidationError):
            result.query = "tampered"  # type: ignore[misc]

    def test_search_hit_is_frozen(self) -> None:
        hit = SearchHit(title="t", href="https://example.com", body="b")
        with pytest.raises(ValidationError):
            hit.title = "changed"  # type: ignore[misc]
