"""Persistent memory exposed as a stdio MCP server.

WAT layer: **Tool** — deterministic execution with no reasoning of its
own. The module backs a tiny long-term memory store with local SQLite and
exposes two tools over the MCP ``@mcp.tool()`` decorator:

* ``store_fact(entity, fact)``     — persist a fact with a timestamp.
* ``retrieve_context(query)``      — recall facts matching a query.

It runs as its own process: the ``MemoryMCPClient`` spawns it over stdio,
so ``main()`` simply starts the server on the stdio transport.

The database location comes from the ``MEMORY_DB_PATH`` environment
variable (injected by the client from typed settings); the sentinel
``":memory:"`` selects an ephemeral in-process database. Importing this
module opens no connection and touches no filesystem — the connection is
created lazily on first tool call — so the test suite exercises the tool
logic offline against an injected in-memory connection. Storage and query
failures are never raised out of a tool; they are returned as strings
prefixed with a clear marker, mirroring the other tools.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Final

from mcp.server.fastmcp import FastMCP

_DB_PATH_ENV: Final[str] = "MEMORY_DB_PATH"
_IN_MEMORY: Final[str] = ":memory:"
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
_DEFAULT_DB_PATH: Final[Path] = _PROJECT_ROOT / ".data" / "memory.db"
_MAX_RESULTS: Final[int] = 10
_MAX_FACT_CHARS: Final[int] = 2_000

mcp = FastMCP(
    "plansmart-memory",
    instructions=(
        "Long-term memory for the assistant. Use `store_fact` to remember "
        "durable facts about people, projects, and preferences, and "
        "`retrieve_context` to recall them before answering."
    ),
)

# Lazily-initialized, process-lifetime connection for the running server.
# Tests never touch this: they inject a connection or patch _get_connection.
_connection: sqlite3.Connection | None = None


def _py_lower(value: str | None) -> str | None:
    """Unicode-aware lower-case for use inside SQL.

    SQLite's built-in ``lower()`` only folds ASCII ``A`` to ``Z``, so a query
    word folded with Python's Unicode ``str.lower()`` would never match an
    accented capital stored verbatim (e.g. Hungarian ``Ágnes``/``KÁVÉT``).
    Registering this on the connection lets both sides fold identically.
    """
    return value.lower() if value is not None else None


def _init_schema(conn: sqlite3.Connection) -> None:
    """Create the ``facts`` table and register helpers on the connection."""
    conn.create_function("pylower", 1, _py_lower, deterministic=True)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity TEXT NOT NULL,
            fact TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()


def _get_connection() -> sqlite3.Connection:
    """Return the process-lifetime SQLite connection, opening it on first use.

    The path is read from ``MEMORY_DB_PATH`` (``:memory:`` selects an
    ephemeral database); a file-backed path has its parent directory
    created on demand.
    """
    global _connection
    if _connection is None:
        raw = os.environ.get(_DB_PATH_ENV, "").strip()
        if raw == _IN_MEMORY:
            _connection = sqlite3.connect(_IN_MEMORY)
        else:
            path = Path(raw) if raw else _DEFAULT_DB_PATH
            path.parent.mkdir(parents=True, exist_ok=True)
            _connection = sqlite3.connect(str(path))
        _init_schema(_connection)
    return _connection


def _store_fact(
    entity: str,
    fact: str,
    *,
    conn: sqlite3.Connection | None = None,
    now: datetime | None = None,
) -> str:
    """Persist ``fact`` about ``entity`` with a timestamp.

    Args:
        entity: The subject the fact concerns (person, project, topic).
        fact: The fact to remember.
        conn: Test-only injection point for a SQLite connection.
        now: Test-only injection point for the timestamp.

    Returns:
        A confirmation string, or an error string prefixed with
        ``"Memory store failed:"`` when the write could not complete.
    """
    entity_clean = entity.strip()
    fact_clean = fact.strip()
    if not entity_clean:
        return "Memory store failed: entity must contain non-whitespace text."
    if not fact_clean:
        return "Memory store failed: fact must contain non-whitespace text."
    fact_clean = fact_clean[:_MAX_FACT_CHARS]
    timestamp = (now or datetime.now()).isoformat()

    try:
        active = conn if conn is not None else _get_connection()
        active.execute(
            "INSERT INTO facts (entity, fact, created_at) VALUES (?, ?, ?)",
            (entity_clean, fact_clean, timestamp),
        )
        active.commit()
    except Exception as exc:
        # A storage failure is surfaced to the model as ordinary tool
        # output, never raised, mirroring the other tools.
        return f"Memory store failed: {type(exc).__name__}: {exc}"

    return f"Stored fact about {entity_clean!r} (at {timestamp})."


def _retrieve_context(
    query: str,
    *,
    conn: sqlite3.Connection | None = None,
    limit: int = _MAX_RESULTS,
) -> str:
    """Return stored facts whose entity or text matches ``query``.

    Matching is a case-insensitive OR over the whitespace-separated words
    of the query, ranked most-recent-first. This is deliberately simple
    keyword recall, not semantic search.

    Args:
        query: Free-text recall query.
        conn: Test-only injection point for a SQLite connection.
        limit: Maximum number of facts to return.

    Returns:
        A formatted digest of matching facts, a "no matches" message, or
        an error string prefixed with ``"Memory retrieval failed:"``.
    """
    words = [word for word in query.lower().split() if word]
    if not words:
        return "Memory retrieval failed: query must contain searchable text."

    conditions = " OR ".join(["pylower(entity) LIKE ? OR pylower(fact) LIKE ?"] * len(words))
    params: list[str | int] = []
    for word in words:
        like = f"%{word}%"
        params.extend([like, like])
    params.append(max(1, limit))
    sql = f"SELECT entity, fact, created_at FROM facts WHERE {conditions} ORDER BY id DESC LIMIT ?"

    try:
        active = conn if conn is not None else _get_connection()
        rows = active.execute(sql, params).fetchall()
    except Exception as exc:
        return f"Memory retrieval failed: {type(exc).__name__}: {exc}"

    if not rows:
        return f"No stored facts match {query!r}."

    lines = [f"Stored facts matching {query!r}:"]
    for index, (entity, fact, created_at) in enumerate(rows, start=1):
        lines.append(f"{index}. [{entity}] {fact} (stored {created_at})")
    return "\n".join(lines)


@mcp.tool()
def store_fact(entity: str, fact: str) -> str:
    """Persist a durable fact to long-term memory with a timestamp.

    Use this whenever the user tells you something worth remembering
    across sessions — a preference, a decision, a deadline, a detail about
    a person or project. Failures come back as a string beginning with
    "Memory store failed:" — never as an exception.

    Args:
        entity: The subject the fact concerns (e.g. a person, project, or
            topic name) — used later as a recall handle.
        fact: The fact to remember, in a self-contained sentence.
    """
    return _store_fact(entity, fact)


@mcp.tool()
def retrieve_context(query: str) -> str:
    """Recall facts from long-term memory that match a query.

    Call this before answering when the user references something they
    told you earlier, or asks what you remember. Returns the most recent
    matching facts (entity, fact, timestamp). Failures come back as a
    string beginning with "Memory retrieval failed:" — never as an
    exception.

    Args:
        query: What to recall; matched by keyword against stored entities
            and facts.
    """
    return _retrieve_context(query)


def main() -> None:
    """Run the memory MCP server on the stdio transport."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
