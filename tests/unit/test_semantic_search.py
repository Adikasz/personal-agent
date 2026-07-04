"""Unit tests for `tools.semantic_search`.

`VectorStore` is injected via the tool's `store` kwarg so no network
I/O occurs. Every code path — schema validation, happy path, empty
match set, and every failure surface (`VectorStoreError`, unexpected
`ValueError`) — is exercised deterministically.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from tools.semantic_search import (
    SemanticMatch,
    SemanticSearchQuery,
    SemanticSearchResult,
    semantic_search,
)
from utils.vector_store import VectorMatch, VectorStoreError


def _fake_store(
    *,
    embedding: list[float] | None = None,
    matches: list[VectorMatch] | None = None,
    embedding_error: Exception | None = None,
    search_error: Exception | None = None,
) -> AsyncMock:
    """Build an AsyncMock that satisfies the VectorStore contract used
    by `semantic_search`."""
    store = AsyncMock()
    if embedding_error is not None:
        store.generate_embedding = AsyncMock(side_effect=embedding_error)
    else:
        store.generate_embedding = AsyncMock(
            return_value=embedding or [0.1, 0.2, 0.3]
        )
    if search_error is not None:
        store.semantic_search = AsyncMock(side_effect=search_error)
    else:
        store.semantic_search = AsyncMock(return_value=matches or [])
    return store


class TestQueryValidation:
    """`SemanticSearchQuery` enforces the LLM-facing input contract."""

    def test_valid_query_round_trips(self) -> None:
        query = SemanticSearchQuery(query="career architect", top_k=3)
        assert query.query == "career architect"
        assert query.top_k == 3

    def test_default_top_k_is_five(self) -> None:
        query = SemanticSearchQuery(query="anything")
        assert query.top_k == 5

    def test_empty_query_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SemanticSearchQuery(query="")

    def test_whitespace_only_query_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SemanticSearchQuery(query="   ")

    def test_overly_long_query_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SemanticSearchQuery(query="x" * 1001)

    def test_top_k_lower_bound(self) -> None:
        with pytest.raises(ValidationError):
            SemanticSearchQuery(query="x", top_k=0)

    def test_top_k_upper_bound(self) -> None:
        with pytest.raises(ValidationError):
            SemanticSearchQuery(query="x", top_k=21)

    def test_extra_fields_are_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            SemanticSearchQuery(query="x", nonsense="drop")  # type: ignore[call-arg]

    def test_query_is_immutable(self) -> None:
        query = SemanticSearchQuery(query="hello")
        with pytest.raises(ValidationError):
            query.query = "changed"  # type: ignore[misc]


class TestHappyPath:
    async def test_returns_matches_populated_from_the_store(self) -> None:
        store = _fake_store(
            embedding=[0.5, 0.6],
            matches=[
                VectorMatch(id="doc-1", score=0.9, metadata={"source": "one.md"}),
                VectorMatch(id="doc-2", score=0.7, metadata={"source": "two.md"}),
            ],
        )
        result = await semantic_search(
            SemanticSearchQuery(query="career growth", top_k=2),
            store=store,
        )

        assert isinstance(result, SemanticSearchResult)
        assert result.query == "career growth"
        assert result.error is None
        assert result.matches == [
            SemanticMatch(id="doc-1", score=0.9, metadata={"source": "one.md"}),
            SemanticMatch(id="doc-2", score=0.7, metadata={"source": "two.md"}),
        ]
        store.generate_embedding.assert_awaited_once_with("career growth")
        store.semantic_search.assert_awaited_once_with([0.5, 0.6], top_k=2)

    async def test_empty_match_set_is_not_an_error(self) -> None:
        store = _fake_store(matches=[])
        result = await semantic_search(
            SemanticSearchQuery(query="something obscure"),
            store=store,
        )
        assert result.matches == []
        assert result.error is None


class TestFailureSurface:
    async def test_vector_store_error_on_embedding_is_captured(self) -> None:
        store = _fake_store(embedding_error=VectorStoreError("OpenAI rate limit"))
        result = await semantic_search(
            SemanticSearchQuery(query="hello"),
            store=store,
        )
        assert result.matches == []
        assert result.error is not None
        assert "OpenAI rate limit" in result.error

    async def test_vector_store_error_on_search_is_captured(self) -> None:
        store = _fake_store(search_error=VectorStoreError("Pinecone unreachable"))
        result = await semantic_search(
            SemanticSearchQuery(query="hello"),
            store=store,
        )
        assert result.matches == []
        assert result.error is not None
        assert "Pinecone unreachable" in result.error

    async def test_value_error_from_store_is_captured(self) -> None:
        store = _fake_store(embedding_error=ValueError("Cannot embed empty text."))
        result = await semantic_search(
            SemanticSearchQuery(query="hello"),
            store=store,
        )
        assert result.matches == []
        assert result.error is not None
        assert "Cannot embed" in result.error


class TestResultShape:
    def test_result_is_frozen(self) -> None:
        result = SemanticSearchResult(query="x", matches=[], error=None)
        with pytest.raises(ValidationError):
            result.query = "changed"  # type: ignore[misc]

    def test_match_is_frozen(self) -> None:
        match = SemanticMatch(id="a", score=0.0, metadata={})
        with pytest.raises(ValidationError):
            match.id = "changed"  # type: ignore[misc]
