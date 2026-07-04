"""Index a document into the Pinecone RAG store.

WAT layer: **Tool** — deterministic execution. The tool computes a
stable content-derived vector id, generates an embedding via OpenAI,
and upserts it to Pinecone together with the caller-supplied metadata.

Every failure inside the vector store is captured on the result's
`error` field so the LLM sees the outage and can pivot without
crashing the agent loop.
"""

from __future__ import annotations

import hashlib
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from utils.logger import get_logger
from utils.vector_store import VectorStore, VectorStoreError

__all__ = [
    "IndexDocumentQuery",
    "IndexDocumentResult",
    "index_document",
]

_logger = get_logger(__name__)

_MAX_CONTENT_LENGTH: Final[int] = 20_000
_MAX_METADATA_KEYS: Final[int] = 20
_CONTENT_PREVIEW_CHARS: Final[int] = 2_000

_STORE: VectorStore | None = None


def _get_store() -> VectorStore:
    """Lazily construct a process-wide `VectorStore`.

    Tests never trigger this path because they always inject a mock
    store via the `store=` kwarg on `index_document`.
    """
    global _STORE
    if _STORE is None:
        _STORE = VectorStore()
    return _STORE


class IndexDocumentQuery(BaseModel):
    """Structured tool input, enforced by pydantic before any API call."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        frozen=True,
    )

    content: str = Field(
        ...,
        min_length=1,
        max_length=_MAX_CONTENT_LENGTH,
        description=(
            "Text body to embed and store. Whitespace-only inputs are "
            "rejected."
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Arbitrary key-value metadata (source URL, filepath, tags, "
            "timestamps) preserved verbatim on the Pinecone record."
        ),
    )

    @field_validator("content")
    @classmethod
    def _content_is_non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(
                "content must contain at least one non-whitespace character"
            )
        return value

    @field_validator("metadata")
    @classmethod
    def _metadata_bounds(cls, values: dict[str, Any]) -> dict[str, Any]:
        if len(values) > _MAX_METADATA_KEYS:
            raise ValueError(
                f"metadata may have at most {_MAX_METADATA_KEYS} keys"
            )
        return values


class IndexDocumentResult(BaseModel):
    """Return payload handed back to the Agent layer."""

    model_config = ConfigDict(frozen=True)

    vector_id: str
    content_length: int
    error: str | None = None


async def index_document(
    query: IndexDocumentQuery,
    *,
    store: VectorStore | None = None,
) -> IndexDocumentResult:
    """Embed `query.content` and upsert it into the Pinecone index.

    The vector id is derived from a SHA-256 hash of the content and its
    source metadata, so re-indexing the same document produces the same
    id (making the operation idempotent) while distinct sources with
    identical text keep separate records.

    Args:
        query: Pre-validated tool input.
        store: Optional dependency-injected `VectorStore`. Production
            callers should leave this `None`; tests supply a mock.

    Returns:
        An `IndexDocumentResult`. Vector-store failures are captured on
        the `error` field, not raised.
    """
    active_store = store or _get_store()
    vector_id = _derive_vector_id(query.content, query.metadata)
    enriched_metadata = _enrich_metadata(query.content, query.metadata)

    try:
        embedding = await active_store.generate_embedding(query.content)
        await active_store.upsert(vector_id, embedding, enriched_metadata)
    except VectorStoreError as exc:
        _logger.warning("index_document failed for %s: %s", vector_id, exc)
        return IndexDocumentResult(
            vector_id=vector_id,
            content_length=len(query.content),
            error=str(exc),
        )
    except ValueError as exc:
        _logger.warning("index_document rejected input for %s: %s", vector_id, exc)
        return IndexDocumentResult(
            vector_id=vector_id,
            content_length=len(query.content),
            error=str(exc),
        )

    _logger.info(
        "index_document indexed %d char(s) as %s.", len(query.content), vector_id
    )
    return IndexDocumentResult(
        vector_id=vector_id,
        content_length=len(query.content),
        error=None,
    )


def _derive_vector_id(content: str, metadata: dict[str, Any]) -> str:
    """Compute a stable content-plus-source vector id."""
    hasher = hashlib.sha256()
    hasher.update(content.encode("utf-8"))
    source = (
        metadata.get("source")
        or metadata.get("url")
        or metadata.get("filepath")
        or ""
    )
    hasher.update(b"\x00")
    hasher.update(str(source).encode("utf-8"))
    return hasher.hexdigest()[:32]


def _enrich_metadata(content: str, metadata: dict[str, Any]) -> dict[str, Any]:
    """Preserve caller metadata verbatim while attaching a content preview.

    Storing a bounded excerpt on the record lets `semantic_search`
    surface human-readable context without a separate document store.
    """
    preview = content[:_CONTENT_PREVIEW_CHARS]
    return {**metadata, "content_preview": preview}
