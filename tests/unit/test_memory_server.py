"""Unit tests for `mcp_integration.memory_server`.

No database file is ever created: helper tests inject an in-memory
`sqlite3.Connection` (the `memory_db` fixture), and tests of the decorated
tools patch the connection factory (the `mock_memory_db` fixture). Both
come from `tests/conftest.py`. Importing the module opens no connection.
"""

from __future__ import annotations

import sqlite3

import pytest

from mcp_integration import memory_server
from mcp_integration.memory_server import (
    _retrieve_context,
    _store_fact,
    retrieve_context,
    store_fact,
)
from tests.conftest import FROZEN_NOW


class TestStoreFact:
    """`_store_fact` validates input and persists a timestamped row."""

    def test_stores_row_and_returns_confirmation(self, memory_db: sqlite3.Connection) -> None:
        out = _store_fact("David", "David founded PlanSmart.", conn=memory_db)
        assert out.startswith("Stored fact about 'David'")

        rows = memory_db.execute("SELECT entity, fact FROM facts").fetchall()
        assert rows == [("David", "David founded PlanSmart.")]

    def test_timestamp_is_injectable_and_recorded(self, memory_db: sqlite3.Connection) -> None:
        _store_fact("David", "likes terse answers", conn=memory_db, now=FROZEN_NOW)
        (created_at,) = memory_db.execute("SELECT created_at FROM facts").fetchone()
        assert created_at == FROZEN_NOW.isoformat()

    def test_blank_entity_is_rejected(self, memory_db: sqlite3.Connection) -> None:
        out = _store_fact("   ", "some fact", conn=memory_db)
        assert out.startswith("Memory store failed:")
        assert "entity" in out
        assert memory_db.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 0

    def test_blank_fact_is_rejected(self, memory_db: sqlite3.Connection) -> None:
        out = _store_fact("David", "   ", conn=memory_db)
        assert out.startswith("Memory store failed:")
        assert "fact" in out

    def test_long_fact_is_clipped(self, memory_db: sqlite3.Connection) -> None:
        _store_fact("Big", "x" * 5000, conn=memory_db)
        (stored,) = memory_db.execute("SELECT fact FROM facts").fetchone()
        assert len(stored) == memory_server._MAX_FACT_CHARS

    def test_storage_error_is_captured_not_raised(self, memory_db: sqlite3.Connection) -> None:
        memory_db.close()  # operating on a closed connection raises
        out = _store_fact("David", "fact", conn=memory_db)
        assert out.startswith("Memory store failed:")


class TestRetrieveContext:
    """`_retrieve_context` performs keyword recall, most-recent-first."""

    def test_empty_query_is_rejected(self, memory_db: sqlite3.Connection) -> None:
        out = _retrieve_context("   ", conn=memory_db)
        assert out.startswith("Memory retrieval failed:")

    def test_matching_facts_are_returned(self, memory_db: sqlite3.Connection) -> None:
        _store_fact("David", "David is the founder of PlanSmart.", conn=memory_db)
        _store_fact("Budapest", "Budapest is the capital of Hungary.", conn=memory_db)

        out = _retrieve_context("david", conn=memory_db)
        assert "founder of PlanSmart" in out
        assert "Budapest" not in out

    def test_no_matches_reports_cleanly(self, memory_db: sqlite3.Connection) -> None:
        _store_fact("David", "David likes espresso.", conn=memory_db)
        out = _retrieve_context("nonexistent", conn=memory_db)
        assert out == "No stored facts match 'nonexistent'."

    def test_matching_is_case_insensitive(self, memory_db: sqlite3.Connection) -> None:
        _store_fact("David", "Founder of PlanSmart.", conn=memory_db)
        assert "PlanSmart" in _retrieve_context("PLANSMART", conn=memory_db)

    def test_accented_uppercase_is_recalled_case_insensitively(
        self, memory_db: sqlite3.Connection
    ) -> None:
        # Hungarian accented capitals must fold symmetrically: SQLite's
        # ASCII-only lower() would otherwise never match a lowercase query.
        _store_fact("Ágnes", "Ágnes szereti a KÁVÉT.", conn=memory_db)
        assert "Ágnes" in _retrieve_context("ágnes", conn=memory_db)
        assert "KÁVÉT" in _retrieve_context("kávét", conn=memory_db)

    def test_multi_word_query_matches_any_word(self, memory_db: sqlite3.Connection) -> None:
        _store_fact("David", "David runs PlanSmart.", conn=memory_db)
        _store_fact("Coffee", "Espresso is a coffee.", conn=memory_db)
        out = _retrieve_context("plansmart espresso", conn=memory_db)
        assert "David runs PlanSmart." in out
        assert "Espresso is a coffee." in out

    def test_results_are_ordered_most_recent_first(self, memory_db: sqlite3.Connection) -> None:
        _store_fact("Topic", "first fact about topic", conn=memory_db)
        _store_fact("Topic", "second fact about topic", conn=memory_db)
        out = _retrieve_context("topic", conn=memory_db)
        assert out.index("second fact") < out.index("first fact")

    def test_limit_caps_the_number_of_results(self, memory_db: sqlite3.Connection) -> None:
        for i in range(5):
            _store_fact("Topic", f"fact number {i} about topic", conn=memory_db)
        out = _retrieve_context("topic", conn=memory_db, limit=2)
        assert out.count("about topic") == 2

    def test_retrieval_error_is_captured_not_raised(self, memory_db: sqlite3.Connection) -> None:
        memory_db.close()
        out = _retrieve_context("anything", conn=memory_db)
        assert out.startswith("Memory retrieval failed:")


class TestToolSurface:
    """The `@mcp.tool()` decorators expose both tools correctly."""

    async def test_both_tools_are_registered_with_expected_schema(self) -> None:
        tools = await memory_server.mcp.list_tools()
        by_name = {tool.name: tool for tool in tools}
        assert "store_fact" in by_name
        assert "retrieve_context" in by_name

        store_props = by_name["store_fact"].inputSchema["properties"]
        assert "entity" in store_props
        assert "fact" in store_props
        assert "query" in by_name["retrieve_context"].inputSchema["properties"]

    def test_decorated_tools_persist_and_recall_via_patched_connection(
        self, mock_memory_db: sqlite3.Connection
    ) -> None:
        stored = store_fact("David", "David prefers Hungarian in chat.")
        assert "Stored fact about 'David'" in stored

        recalled = retrieve_context("David")
        assert "Hungarian" in recalled
        # The patched in-memory connection actually holds the row.
        assert mock_memory_db.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 1


@pytest.mark.parametrize("query", ["", "   ", "\t\n"])
def test_blank_queries_are_uniformly_rejected(query: str, memory_db: sqlite3.Connection) -> None:
    assert _retrieve_context(query, conn=memory_db).startswith("Memory retrieval failed:")
