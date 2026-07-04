"""Asynchronous facade over OpenAI embeddings and Pinecone vector storage.

WAT layer: **Tool** — deterministic infrastructure. This module is the
only place in the codebase that imports `openai` or `pinecone`. Every
Agent tool that needs semantic recall or indexing must call through this
facade so the specifics of either vendor stay swappable.

Design decisions worth calling out:

- `VectorStore` is a synchronous constructor. Both underlying vendor
  clients are created eagerly; construction failures surface early as
  standard exceptions, not lazy at first use.
- `generate_embedding` uses the OpenAI async client directly (network
  I/O is already awaitable there).
- `semantic_search` and `upsert` dispatch the blocking Pinecone SDK
  calls through `asyncio.to_thread` so the assistant's event loop is
  never stalled.
- Every fault path — rate limits, network errors, malformed responses —
  is captured by this module's `VectorStoreError`. Downstream Agent
  tools translate that exception into a structured `error` field on
  their own pydantic result, matching the pattern already established
  by `tools/web_search.py` and `tools/scrape_url.py`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI
from pinecone import Pinecone

from core.config import Settings, get_settings
from utils.logger import get_logger

__all__ = [
    "VectorMatch",
    "VectorStore",
    "VectorStoreError",
]


class VectorStoreError(Exception):
    """Raised when OpenAI or Pinecone signals a fault the tool must handle."""


@dataclass(frozen=True)
class VectorMatch:
    """A single match returned by `VectorStore.semantic_search`."""

    id: str
    score: float
    metadata: dict[str, Any]


class VectorStore:
    """Async facade over OpenAI embeddings + Pinecone vector storage."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        openai_client: AsyncOpenAI | None = None,
        pinecone_client: Any | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._logger = get_logger(self.__class__.__name__)
        self._openai = openai_client or AsyncOpenAI(
            api_key=self._settings.openai_api_key.get_secret_value()
        )
        self._pinecone = pinecone_client or Pinecone(
            api_key=self._settings.pinecone_api_key.get_secret_value()
        )
        self._index = self._pinecone.Index(self._settings.pinecone_index_name)

    async def generate_embedding(self, text: str) -> list[float]:
        """Return the embedding vector for `text`.

        Raises:
            ValueError: If `text` is empty or whitespace only.
            VectorStoreError: If the OpenAI API rejects the call for any
                reason (auth, rate limit, network, malformed response).
        """
        if not text.strip():
            raise ValueError("Cannot embed empty text.")

        try:
            response = await self._openai.embeddings.create(
                model=self._settings.openai_embedding_model,
                input=text,
            )
        except Exception as exc:  # noqa: BLE001 — every OpenAI fault becomes VectorStoreError
            self._logger.warning("OpenAI embedding failed: %s", exc)
            raise VectorStoreError(f"OpenAI embedding failed: {exc}") from exc

        try:
            vector = list(response.data[0].embedding)
        except (AttributeError, IndexError, TypeError) as exc:
            self._logger.warning("OpenAI response was malformed: %s", exc)
            raise VectorStoreError(
                f"OpenAI response was malformed: {exc}"
            ) from exc

        self._logger.info("Generated %d-dimensional embedding.", len(vector))
        return vector

    async def semantic_search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 5,
    ) -> list[VectorMatch]:
        """Return the top-`top_k` Pinecone matches for `query_embedding`.

        Raises:
            VectorStoreError: If the Pinecone API rejects the call.
        """
        try:
            response = await asyncio.to_thread(
                self._index.query,
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
            )
        except Exception as exc:  # noqa: BLE001 — Pinecone faults become VectorStoreError
            self._logger.warning("Pinecone query failed: %s", exc)
            raise VectorStoreError(f"Pinecone query failed: {exc}") from exc

        matches: list[VectorMatch] = []
        for match in _iter_matches(response):
            matches.append(
                VectorMatch(
                    id=str(match.get("id", "")),
                    score=float(match.get("score", 0.0)),
                    metadata=dict(match.get("metadata") or {}),
                )
            )

        self._logger.info("Pinecone returned %d match(es).", len(matches))
        return matches

    async def upsert(
        self,
        vector_id: str,
        embedding: list[float],
        metadata: dict[str, Any],
    ) -> None:
        """Insert or update a single vector in Pinecone.

        Raises:
            VectorStoreError: If the Pinecone API rejects the call.
        """
        payload = [
            {
                "id": vector_id,
                "values": embedding,
                "metadata": metadata,
            }
        ]
        try:
            await asyncio.to_thread(self._index.upsert, vectors=payload)
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("Pinecone upsert failed: %s", exc)
            raise VectorStoreError(f"Pinecone upsert failed: {exc}") from exc

        self._logger.info("Upserted vector %s.", vector_id)


def _iter_matches(response: Any) -> list[dict[str, Any]]:
    """Normalize a Pinecone query response into a list of dict-like matches.

    Different Pinecone client versions return either an object with a
    `.matches` attribute (whose items expose `.id`, `.score`, and
    `.metadata`) or a plain dict-like response. This helper accepts
    both, so unit tests can supply the simpler dict shape as fixtures.
    """
    raw: Any
    if hasattr(response, "matches"):
        raw = response.matches
    elif isinstance(response, dict):
        raw = response.get("matches", [])
    else:
        raw = []

    normalized: list[dict[str, Any]] = []
    for match in raw:
        if isinstance(match, dict):
            normalized.append(match)
        else:
            normalized.append(
                {
                    "id": getattr(match, "id", ""),
                    "score": getattr(match, "score", 0.0),
                    "metadata": getattr(match, "metadata", {}),
                }
            )
    return normalized
