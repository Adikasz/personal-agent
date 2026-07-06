"""Unit tests for `utils.vector_store`.

Both the OpenAI and Pinecone clients are mocked at the class level so
the test suite runs offline. Every code path — success, empty input
guard, malformed OpenAI response, OpenAI error, Pinecone error,
match normalization across response shapes — is exercised
deterministically.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from utils.vector_store import VectorMatch, VectorStore, VectorStoreError


def _build_store(
    *,
    openai_client: AsyncMock | None = None,
    pinecone_index: MagicMock | None = None,
) -> VectorStore:
    """Instantiate `VectorStore` with fully controlled vendor doubles."""
    fake_openai = openai_client or AsyncMock()
    fake_index = pinecone_index or MagicMock()
    fake_pinecone = MagicMock()
    fake_pinecone.Index.return_value = fake_index
    return VectorStore(
        openai_client=fake_openai,
        pinecone_client=fake_pinecone,
    )


class TestGenerateEmbedding:
    """OpenAI embedding path."""

    async def test_returns_embedding_vector_on_success(self) -> None:
        fake_openai = AsyncMock()
        fake_openai.embeddings.create = AsyncMock(
            return_value=SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])])
        )
        store = _build_store(openai_client=fake_openai)

        vector = await store.generate_embedding("hello world")

        assert vector == [0.1, 0.2, 0.3]
        fake_openai.embeddings.create.assert_awaited_once()
        call_kwargs = fake_openai.embeddings.create.await_args.kwargs
        assert call_kwargs["input"] == "hello world"
        assert call_kwargs["model"] == "text-embedding-3-small"

    async def test_empty_input_raises_value_error(self) -> None:
        store = _build_store()
        with pytest.raises(ValueError):
            await store.generate_embedding("")

    async def test_whitespace_input_raises_value_error(self) -> None:
        store = _build_store()
        with pytest.raises(ValueError):
            await store.generate_embedding("   ")

    async def test_openai_exception_is_wrapped_as_vector_store_error(self) -> None:
        fake_openai = AsyncMock()
        fake_openai.embeddings.create = AsyncMock(side_effect=RuntimeError("429 Too Many Requests"))
        store = _build_store(openai_client=fake_openai)
        with pytest.raises(VectorStoreError) as exc_info:
            await store.generate_embedding("hello")
        assert "429" in str(exc_info.value)

    async def test_malformed_openai_response_is_wrapped(self) -> None:
        fake_openai = AsyncMock()
        # `data` list is empty, so `data[0]` raises IndexError.
        fake_openai.embeddings.create = AsyncMock(return_value=SimpleNamespace(data=[]))
        store = _build_store(openai_client=fake_openai)
        with pytest.raises(VectorStoreError):
            await store.generate_embedding("hello")


class TestSemanticSearch:
    """Pinecone query path."""

    async def test_returns_matches_from_object_shape(self) -> None:
        fake_index = MagicMock()
        fake_index.query.return_value = SimpleNamespace(
            matches=[
                SimpleNamespace(id="doc-1", score=0.98, metadata={"source": "notes/one.md"}),
                SimpleNamespace(id="doc-2", score=0.71, metadata={"source": "notes/two.md"}),
            ]
        )
        store = _build_store(pinecone_index=fake_index)

        matches = await store.semantic_search([0.1, 0.2], top_k=3)

        assert matches == [
            VectorMatch(id="doc-1", score=0.98, metadata={"source": "notes/one.md"}),
            VectorMatch(id="doc-2", score=0.71, metadata={"source": "notes/two.md"}),
        ]
        fake_index.query.assert_called_once_with(vector=[0.1, 0.2], top_k=3, include_metadata=True)

    async def test_returns_matches_from_dict_shape(self) -> None:
        fake_index = MagicMock()
        fake_index.query.return_value = {
            "matches": [{"id": "d", "score": 0.5, "metadata": {"tag": "career"}}]
        }
        store = _build_store(pinecone_index=fake_index)

        matches = await store.semantic_search([0.0], top_k=1)
        assert matches == [VectorMatch(id="d", score=0.5, metadata={"tag": "career"})]

    async def test_missing_matches_field_yields_empty_list(self) -> None:
        fake_index = MagicMock()
        fake_index.query.return_value = {"anything": "else"}
        store = _build_store(pinecone_index=fake_index)
        assert await store.semantic_search([0.1]) == []

    async def test_unrecognized_response_shape_yields_empty_list(self) -> None:
        fake_index = MagicMock()
        fake_index.query.return_value = 42  # neither object nor dict
        store = _build_store(pinecone_index=fake_index)
        assert await store.semantic_search([0.1]) == []

    async def test_pinecone_exception_is_wrapped_as_vector_store_error(self) -> None:
        fake_index = MagicMock()
        fake_index.query.side_effect = ConnectionError("pinecone unreachable")
        store = _build_store(pinecone_index=fake_index)
        with pytest.raises(VectorStoreError) as exc_info:
            await store.semantic_search([0.1])
        assert "pinecone unreachable" in str(exc_info.value)

    async def test_metadata_none_is_normalized_to_empty_dict(self) -> None:
        fake_index = MagicMock()
        fake_index.query.return_value = {"matches": [{"id": "x", "score": 0.1, "metadata": None}]}
        store = _build_store(pinecone_index=fake_index)
        matches = await store.semantic_search([0.1])
        assert matches[0].metadata == {}


class TestUpsert:
    """Pinecone upsert path."""

    async def test_upsert_forwards_payload_to_pinecone(self) -> None:
        fake_index = MagicMock()
        store = _build_store(pinecone_index=fake_index)

        await store.upsert("id-1", [0.1, 0.2], {"source": "notes/a.md"})

        fake_index.upsert.assert_called_once_with(
            vectors=[
                {
                    "id": "id-1",
                    "values": [0.1, 0.2],
                    "metadata": {"source": "notes/a.md"},
                }
            ]
        )

    async def test_upsert_exception_is_wrapped_as_vector_store_error(self) -> None:
        fake_index = MagicMock()
        fake_index.upsert.side_effect = RuntimeError("write forbidden")
        store = _build_store(pinecone_index=fake_index)
        with pytest.raises(VectorStoreError) as exc_info:
            await store.upsert("id-1", [0.1], {})
        assert "write forbidden" in str(exc_info.value)


class TestConstruction:
    """Default vendor-client construction uses the injected settings."""

    def test_default_constructor_wires_settings_derived_clients(self) -> None:
        with (
            patch("utils.vector_store.AsyncOpenAI") as fake_openai_cls,
            patch("utils.vector_store.Pinecone") as fake_pinecone_cls,
        ):
            store = VectorStore()

        fake_openai_cls.assert_called_once()
        fake_pinecone_cls.assert_called_once()
        assert store._index is fake_pinecone_cls.return_value.Index.return_value


def _sanity_dataclass_shape() -> Any:
    """Sanity check that `VectorMatch` is a frozen dataclass."""
    return VectorMatch(id="a", score=0.0, metadata={})


class TestVectorMatch:
    def test_vector_match_is_frozen(self) -> None:
        match = _sanity_dataclass_shape()
        with pytest.raises(FrozenInstanceError):
            match.id = "changed"
