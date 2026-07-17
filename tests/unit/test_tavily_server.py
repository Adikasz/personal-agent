"""Unit tests for `mcp_integration.tavily_server`.

The `tavily-python` SDK is never exercised against the network: the search
logic accepts an injected fake client, and the one test that calls the
decorated tool patches the internal `_search` helper. Importing the module
constructs no client and performs no I/O.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from mcp_integration import tavily_server
from mcp_integration.tavily_server import (
    DEFAULT_MAX_RESULTS,
    _search,
    tavily_search,
)


class _FakeTavily:
    """A stand-in for `tavily.TavilyClient` that records its `search` call."""

    def __init__(self, payload: Any, *, record: dict[str, Any] | None = None) -> None:
        self._payload = payload
        self._record = record

    def search(self, query: str, **kwargs: Any) -> Any:
        if self._record is not None:
            self._record.update({"query": query, **kwargs})
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class TestSearchValidation:
    """Guard clauses run before any client is constructed."""

    def test_blank_query_is_rejected_without_touching_the_client(self) -> None:
        result = _search("   ", 5, client=_FakeTavily({"results": []}))
        assert result.startswith("Tavily search failed:")
        assert "non-whitespace" in result

    def test_max_results_is_clamped_to_upper_bound(self) -> None:
        record: dict[str, Any] = {}
        _search("q", 100, client=_FakeTavily({"results": []}, record=record))
        assert record["max_results"] == 20

    def test_max_results_is_clamped_to_lower_bound(self) -> None:
        record: dict[str, Any] = {}
        _search("q", 0, client=_FakeTavily({"results": []}, record=record))
        assert record["max_results"] == 1

    def test_search_parameters_are_forwarded(self) -> None:
        record: dict[str, Any] = {}
        _search("careers", 3, client=_FakeTavily({"results": []}, record=record))
        assert record["query"] == "careers"
        assert record["max_results"] == 3
        assert record["include_answer"] is True
        assert record["search_depth"] == "basic"


class TestSearchFormatting:
    """A well-formed Tavily payload becomes an LLM-ready digest."""

    def test_answer_and_results_are_rendered(self) -> None:
        payload = {
            "answer": "Paris is the capital of France.",
            "results": [
                {
                    "title": "France",
                    "url": "https://example.com/france",
                    "content": "France is a country in Western Europe.",
                },
                {
                    "title": "Paris",
                    "url": "https://example.com/paris",
                    "content": "Paris is the capital and largest city.",
                },
            ],
        }
        out = _search("capital of France", 5, client=_FakeTavily(payload))
        assert "Search results for: capital of France" in out
        assert "Summary: Paris is the capital of France." in out
        assert "1. France" in out
        assert "URL: https://example.com/france" in out
        assert "2. Paris" in out

    def test_long_content_is_clipped(self) -> None:
        payload = {
            "results": [{"title": "Big", "url": "https://example.com/big", "content": "x" * 1000}]
        }
        out = _search("q", 1, client=_FakeTavily(payload))
        assert "…" in out
        assert "x" * 1000 not in out

    def test_empty_results_are_reported_without_error_prefix(self) -> None:
        out = _search("obscure", 5, client=_FakeTavily({"results": []}))
        assert "No results were returned" in out
        assert not out.startswith("Tavily search failed:")

    def test_missing_answer_omits_summary_line(self) -> None:
        payload = {"results": [{"title": "T", "url": "https://example.com", "content": "c"}]}
        out = _search("q", 1, client=_FakeTavily(payload))
        assert "Summary:" not in out


class TestSearchFailureHandling:
    """Failures are surfaced as strings, never raised."""

    def test_client_exception_is_captured_in_the_return_string(self) -> None:
        out = _search("q", 5, client=_FakeTavily(RuntimeError("boom")))
        assert out.startswith("Tavily search failed:")
        assert "RuntimeError" in out
        assert "boom" in out

    def test_non_dict_payload_is_reported(self) -> None:
        out = _search("q", 5, client=_FakeTavily(["not", "a", "dict"]))
        assert out.startswith("Tavily search failed:")
        assert "unexpected response shape" in out


class TestToolSurface:
    """The `@mcp.tool()` decorator exposes `tavily_search` correctly."""

    async def test_tool_is_registered_with_expected_schema(self) -> None:
        tools = await tavily_server.mcp.list_tools()
        by_name = {tool.name: tool for tool in tools}
        assert "tavily_search" in by_name
        properties = by_name["tavily_search"].inputSchema["properties"]
        assert "query" in properties
        assert "max_results" in properties

    def test_tool_delegates_to_search_helper(self) -> None:
        with patch.object(tavily_server, "_search", return_value="stub") as search_mock:
            assert tavily_search("hello", 3) == "stub"
        search_mock.assert_called_once_with("hello", 3)

    def test_default_max_results_constant(self) -> None:
        assert DEFAULT_MAX_RESULTS == 5


@pytest.mark.parametrize("query", ["", "   ", "\t\n"])
def test_blank_queries_are_uniformly_rejected(query: str) -> None:
    assert _search(query, 5, client=_FakeTavily({"results": []})).startswith(
        "Tavily search failed:"
    )
