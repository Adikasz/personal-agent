"""Semantic search over the Pinecone RAG store.

WAT layer: **Tool** — deterministic execution. The tool composes two
`VectorStore` calls (embed the query, then query Pinecone) and returns
a strongly-typed pydantic result. Every failure inside the vector store
is captured on the result's `error` field so the LLM sees the outage
and can pivot without crashing the agent loop.
"""

from __future__ import annotations

from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from utils.logger import get_logger
from utils.vector_store import VectorStore, VectorStoreError

__all__ = [
    "SemanticMatch",
    "SemanticSearchQuery",
    "SemanticSearchResult",
    "semantic_search",
]

_logger = get_logger(__name__)

_MAX_QUERY_LENGTH: Final[int] = 1_000
_DEFAULT_TOP_K: Final[int] = 5
_MAX_TOP_K: Final[int] = 20

_STORE: VectorStore | None = None


def _get_store() -> VectorStore:
    """Lazily construct a process-wide `VectorStore`.

    Tests never trigger this path because they always inject a mock
    store via the `store=` kwarg on `semantic_search`.
    """
    global _STORE
    if _STORE is None:
        _STORE = VectorStore()
    return _STORE


class SemanticSearchQuery(BaseModel):
    """Structured tool input, enforced by pydantic before any API call."""

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
            "Natural-language query embedded and matched against the Pinecone vector store."
        ),
    )
    top_k: int = Field(
        default=_DEFAULT_TOP_K,
        ge=1,
        le=_MAX_TOP_K,
        description="Upper bound on the number of matches to return.",
    )

    @field_validator("query")
    @classmethod
    def _query_is_non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must contain at least one non-whitespace character")
        return value


class SemanticMatch(BaseModel):
    """A single retrieved document with its similarity score."""

    model_config = ConfigDict(frozen=True)

    id: str
    score: float
    metadata: dict[str, Any]


class SemanticSearchResult(BaseModel):
    """Return payload handed back to the Agent layer.

    On success the tool populates `matches`; `error` is `None`. On
    failure the tool populates `error` with the human-readable message
    from `VectorStoreError` and leaves `matches` empty.
    """

    model_config = ConfigDict(frozen=True)

    query: str
    matches: list[SemanticMatch]
    error: str | None = None


async def semantic_search(
    query: SemanticSearchQuery,
    *,
    store: VectorStore | None = None,
) -> SemanticSearchResult:
    """Embed `query.query` and return the top matches from Pinecone.

    Args:
        query: Pre-validated tool input.
        store: Optional dependency-injected `VectorStore`. Production
            callers should leave this `None`; tests supply a mock.

    Returns:
        A `SemanticSearchResult`. Vector-store failures are captured on
        the `error` field, not raised.
    """
    active_store = store or _get_store()
    try:
        embedding = await active_store.generate_embedding(query.query)
        raw_matches = await active_store.semantic_search(embedding, top_k=query.top_k)
    except VectorStoreError as exc:
        _logger.warning("semantic_search failed for %r: %s", query.query, exc)
        return SemanticSearchResult(query=query.query, matches=[], error=str(exc))
    except ValueError as exc:
        _logger.warning("semantic_search rejected input %r: %s", query.query, exc)
        return SemanticSearchResult(query=query.query, matches=[], error=str(exc))

    matches = [
        SemanticMatch(id=match.id, score=match.score, metadata=match.metadata)
        for match in raw_matches
    ]
    _logger.info(
        "semantic_search returned %d match(es) for %r.",
        len(matches),
        query.query,
    )
    return SemanticSearchResult(query=query.query, matches=matches, error=None)
