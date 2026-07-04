"""Unit tests for `tools.index_document`.

`VectorStore` is injected via the tool's `store` kwarg so no network
I/O occurs. Every code path — schema validation, happy path,
deterministic id derivation, content preview attachment, and every
failure surface (`VectorStoreError`, `ValueError`) — is exercised
deterministically.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from tools.index_document import (
    IndexDocumentQuery,
    IndexDocumentResult,
    index_document,
)
from utils.vector_store import VectorStoreError


def _fake_store(
    *,
    embedding: list[float] | None = None,
    embedding_error: Exception | None = None,
    upsert_error: Exception | None = None,
) -> AsyncMock:
    """Build an AsyncMock that satisfies the VectorStore contract used
    by `index_document`."""
    store = AsyncMock()
    if embedding_error is not None:
        store.generate_embedding = AsyncMock(side_effect=embedding_error)
    else:
        store.generate_embedding = AsyncMock(
            return_value=embedding or [0.1, 0.2, 0.3]
        )
    if upsert_error is not None:
        store.upsert = AsyncMock(side_effect=upsert_error)
    else:
        store.upsert = AsyncMock(return_value=None)
    return store


class TestQueryValidation:
    """`IndexDocumentQuery` enforces a strict, minimal input contract."""

    def test_valid_query_round_trips(self) -> None:
        query = IndexDocumentQuery(
            content="body", metadata={"source": "notes/x.md"}
        )
        assert query.content == "body"
        assert query.metadata == {"source": "notes/x.md"}

    def test_default_metadata_is_empty(self) -> None:
        query = IndexDocumentQuery(content="body")
        assert query.metadata == {}

    def test_empty_content_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IndexDocumentQuery(content="")

    def test_whitespace_only_content_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IndexDocumentQuery(content="   ")

    def test_overly_long_content_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IndexDocumentQuery(content="x" * 20_001)

    def test_too_many_metadata_keys_are_rejected(self) -> None:
        oversized = {f"k{i}": i for i in range(21)}
        with pytest.raises(ValidationError):
            IndexDocumentQuery(content="body", metadata=oversized)

    def test_extra_fields_are_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            IndexDocumentQuery(content="body", nonsense="drop")  # type: ignore[call-arg]

    def test_query_is_immutable(self) -> None:
        query = IndexDocumentQuery(content="body")
        with pytest.raises(ValidationError):
            query.content = "changed"  # type: ignore[misc]


class TestHappyPath:
    async def test_indexes_content_and_reports_metadata(self) -> None:
        store = _fake_store(embedding=[0.4, 0.5])

        result = await index_document(
            IndexDocumentQuery(
                content="career architect notes",
                metadata={"source": "notes/career.md"},
            ),
            store=store,
        )

        assert isinstance(result, IndexDocumentResult)
        assert result.error is None
        assert result.content_length == len("career architect notes")
        assert len(result.vector_id) == 32
        store.generate_embedding.assert_awaited_once_with(
            "career architect notes"
        )
        store.upsert.assert_awaited_once()
        upsert_args = store.upsert.await_args
        assert upsert_args.args[0] == result.vector_id
        assert upsert_args.args[1] == [0.4, 0.5]
        stored_metadata = upsert_args.args[2]
        assert stored_metadata["source"] == "notes/career.md"
        assert stored_metadata["content_preview"] == "career architect notes"

    async def test_identical_input_produces_identical_vector_id(self) -> None:
        store_a = _fake_store()
        store_b = _fake_store()
        result_a = await index_document(
            IndexDocumentQuery(
                content="same body", metadata={"source": "notes/x.md"}
            ),
            store=store_a,
        )
        result_b = await index_document(
            IndexDocumentQuery(
                content="same body", metadata={"source": "notes/x.md"}
            ),
            store=store_b,
        )
        assert result_a.vector_id == result_b.vector_id

    async def test_different_source_produces_different_vector_id(self) -> None:
        store_a = _fake_store()
        store_b = _fake_store()
        result_a = await index_document(
            IndexDocumentQuery(
                content="same body", metadata={"source": "notes/one.md"}
            ),
            store=store_a,
        )
        result_b = await index_document(
            IndexDocumentQuery(
                content="same body", metadata={"source": "notes/two.md"}
            ),
            store=store_b,
        )
        assert result_a.vector_id != result_b.vector_id

    async def test_source_falls_back_across_url_and_filepath_keys(self) -> None:
        store_url = _fake_store()
        store_filepath = _fake_store()

        result_url = await index_document(
            IndexDocumentQuery(
                content="body", metadata={"url": "https://example.com"}
            ),
            store=store_url,
        )
        result_filepath = await index_document(
            IndexDocumentQuery(
                content="body", metadata={"filepath": "notes/x.md"}
            ),
            store=store_filepath,
        )
        # Different sources → different ids.
        assert result_url.vector_id != result_filepath.vector_id

    async def test_content_preview_is_capped_at_configured_length(self) -> None:
        long_content = "x" * 5_000
        store = _fake_store()
        await index_document(
            IndexDocumentQuery(content=long_content), store=store
        )
        stored_metadata = store.upsert.await_args.args[2]
        assert len(stored_metadata["content_preview"]) == 2_000


class TestFailureSurface:
    async def test_vector_store_error_on_embedding_is_captured(self) -> None:
        store = _fake_store(embedding_error=VectorStoreError("OpenAI down"))
        result = await index_document(
            IndexDocumentQuery(content="body"), store=store
        )
        assert result.error is not None
        assert "OpenAI down" in result.error
        assert result.content_length == len("body")

    async def test_vector_store_error_on_upsert_is_captured(self) -> None:
        store = _fake_store(upsert_error=VectorStoreError("Pinecone quota"))
        result = await index_document(
            IndexDocumentQuery(content="body"), store=store
        )
        assert result.error is not None
        assert "Pinecone quota" in result.error

    async def test_value_error_from_store_is_captured(self) -> None:
        store = _fake_store(embedding_error=ValueError("Cannot embed"))
        result = await index_document(
            IndexDocumentQuery(content="body"), store=store
        )
        assert result.error is not None
        assert "Cannot embed" in result.error


class TestResultShape:
    def test_result_is_frozen(self) -> None:
        result = IndexDocumentResult(
            vector_id="abc", content_length=3, error=None
        )
        with pytest.raises(ValidationError):
            result.vector_id = "changed"  # type: ignore[misc]
