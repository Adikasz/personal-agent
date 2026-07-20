"""Shared fixtures for the PlanSmart test suite.

Every test starts from a clean state: the cached `Settings` instance is
cleared, any leftover environment variables that could leak from `.env`
into the process are removed, and sensible dummy values are injected
for the three required API keys. Tests that specifically want to
exercise the missing-key path can override the defaults with
`monkeypatch.delenv(...)`.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence
from datetime import datetime, tzinfo
from types import SimpleNamespace
from typing import Any, Self
from unittest.mock import AsyncMock

import pytest

from core.config import get_settings
from mcp_integration.types import MCPToolError, MCPToolInfo

#: The single timestamp every test observes. Tests that assert on the
#: temporal anchor import this constant instead of hard-coding strings.
FROZEN_NOW: datetime = datetime(2026, 7, 17, 12, 0, 0)


class _FrozenDatetime(datetime):
    """`datetime` subclass whose `now()` is pinned to `FROZEN_NOW`.

    Patched over `agents.personal_assistant.datetime` so the temporal
    anchor injected into the system prompt is deterministic. Subclassing
    (rather than a bare stub) keeps every other `datetime` behavior —
    arithmetic, formatting, comparisons — fully intact for any code that
    receives one of these instances.
    """

    @classmethod
    def now(cls, tz: tzinfo | None = None) -> Self:
        return cls(
            FROZEN_NOW.year,
            FROZEN_NOW.month,
            FROZEN_NOW.day,
            FROZEN_NOW.hour,
            FROZEN_NOW.minute,
            FROZEN_NOW.second,
            tzinfo=tz,
        )


@pytest.fixture(autouse=True)
def _frozen_datetime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Freeze `datetime.now()` inside the assistant module for every test.

    The patch is applied at the consuming module's seam
    (`agents.personal_assistant.datetime`), not globally, so stdlib and
    third-party code keep the real clock while the temporal anchor stays
    deterministic under the suite's warnings-as-errors regime.
    """
    monkeypatch.setattr("agents.personal_assistant.datetime", _FrozenDatetime)


_ENV_VARS_UNDER_TEST: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_MAX_TOKENS",
    "MAX_HISTORY_TURNS",
    "MAX_TOOL_ITERATIONS",
    "LOG_LEVEL",
    "KNOWLEDGE_DIR",
    "OPENAI_API_KEY",
    "OPENAI_EMBEDDING_MODEL",
    "PINECONE_API_KEY",
    "PINECONE_INDEX_NAME",
    "TAVILY_API_KEY",
    "MEMORY_DB_PATH",
)

_INJECTED_DEFAULTS: dict[str, str] = {
    # Dummies for the three required secret keys so tests unrelated to the
    # config layer can construct `Settings` (directly or via `get_settings`)
    # without any real credentials. Every key `Settings` marks as required
    # must appear here, otherwise the suite silently depends on a local
    # `.env` and breaks on a CI runner that has neither the file nor the
    # variables exported. Tests that specifically exercise the missing-key
    # path override these with `monkeypatch.delenv(...)`.
    "ANTHROPIC_API_KEY": "sk-ant-test",
    "OPENAI_API_KEY": "sk-test-openai",
    "PINECONE_API_KEY": "pc-test-pinecone",
    "TAVILY_API_KEY": "tvly-test-key",
}


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Guarantee a fresh `Settings` cache and a clean environment per test."""
    for var in _ENV_VARS_UNDER_TEST:
        monkeypatch.delenv(var, raising=False)
    for var, value in _INJECTED_DEFAULTS.items():
        monkeypatch.setenv(var, value)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# MCP test doubles
#
# The offline-CI contract forbids real subprocesses and network I/O, so the
# MCP transport is mocked at two levels:
#
#   * `FakeMCPProvider` — an in-memory `MCPToolProvider` used to exercise the
#     Agent-layer integration (tool discovery + routed execution) with no
#     `mcp` SDK involvement at all.
#   * `mock_mcp_transport` — patches `stdio_client` and `ClientSession` inside
#     `mcp_integration.client` so `TavilyMCPClient` can "connect" without ever
#     spawning `tavily_server.py`.
# ---------------------------------------------------------------------------


def make_tool_info(
    name: str = "tavily_search",
    description: str = "Search the web via Tavily.",
    input_schema: dict[str, Any] | None = None,
) -> MCPToolInfo:
    """Build an `MCPToolInfo` with a sensible default JSON schema."""
    schema = input_schema or {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer"},
        },
        "required": ["query"],
    }
    return MCPToolInfo(name=name, description=description, input_schema=schema)


class FakeMCPProvider:
    """In-memory `MCPToolProvider`: no subprocess, no network, no SDK.

    Records every `call_tool` invocation on `.calls` so tests can assert
    that the assistant routed the right arguments to the right tool.
    """

    def __init__(
        self,
        tools: Sequence[MCPToolInfo] | None = None,
        responses: dict[str, str] | None = None,
        *,
        errors: Sequence[str] = (),
    ) -> None:
        self._tools: list[MCPToolInfo] = list(tools) if tools is not None else [make_tool_info()]
        self._responses: dict[str, str] = dict(responses or {})
        self._errors: set[str] = set(errors)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> list[MCPToolInfo]:
        return list(self._tools)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((name, dict(arguments)))
        if name in self._errors:
            raise MCPToolError(f"MCP tool {name!r} failed (simulated).")
        return self._responses.get(name, f"{name} result")


class _AsyncCM:
    """Minimal async context manager yielding a preset value."""

    def __init__(self, value: Any) -> None:
        self._value = value

    async def __aenter__(self) -> Any:
        return self._value

    async def __aexit__(self, *exc_info: object) -> bool:
        return False


def make_call_tool_result(text: str = "", *, is_error: bool = False) -> SimpleNamespace:
    """Build a `CallToolResult`-shaped double with a single text block."""
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        isError=is_error,
    )


@pytest.fixture
def fake_mcp_provider() -> FakeMCPProvider:
    """A ready-to-use in-memory MCP provider advertising `tavily_search`."""
    return FakeMCPProvider(responses={"tavily_search": "Search results for: test"})


@pytest.fixture
def mcp_session() -> SimpleNamespace:
    """A `ClientSession`-shaped double with stubbed async methods."""
    return SimpleNamespace(
        initialize=AsyncMock(return_value=SimpleNamespace()),
        list_tools=AsyncMock(return_value=SimpleNamespace(tools=[])),
        call_tool=AsyncMock(return_value=make_call_tool_result()),
    )


@pytest.fixture
def mock_mcp_transport(
    monkeypatch: pytest.MonkeyPatch,
    mcp_session: SimpleNamespace,
) -> SimpleNamespace:
    """Patch `stdio_client` + `ClientSession` so `TavilyMCPClient` connects offline.

    Returns the `mcp_session` double so tests can program `list_tools` /
    `call_tool` return values and assert on the calls made through it.
    """
    read_stream = SimpleNamespace(kind="read")
    write_stream = SimpleNamespace(kind="write")

    def fake_stdio_client(server_params: Any, *args: Any, **kwargs: Any) -> _AsyncCM:
        return _AsyncCM((read_stream, write_stream))

    def fake_client_session(read: Any, write: Any, *args: Any, **kwargs: Any) -> _AsyncCM:
        return _AsyncCM(mcp_session)

    monkeypatch.setattr("mcp_integration.client.stdio_client", fake_stdio_client)
    monkeypatch.setattr("mcp_integration.client.ClientSession", fake_client_session)
    return mcp_session


# ---------------------------------------------------------------------------
# Memory-server SQLite doubles
#
# The offline-CI contract forbids real database files, so the persistent-
# memory server is exercised against an in-process SQLite database:
#
#   * `memory_db` — a schema-initialized `sqlite3.Connection` to ":memory:",
#     for tests that inject a connection directly.
#   * `mock_memory_db` — patches `mcp_integration.memory_server._get_connection`
#     to return that same connection, so the decorated MCP tools run against
#     the in-memory database with no filesystem access.
# ---------------------------------------------------------------------------


@pytest.fixture
def memory_db() -> Iterator[sqlite3.Connection]:
    """Yield a schema-initialized in-memory SQLite connection."""
    from mcp_integration.memory_server import _init_schema

    conn = sqlite3.connect(":memory:")
    _init_schema(conn)
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def mock_memory_db(
    monkeypatch: pytest.MonkeyPatch,
    memory_db: sqlite3.Connection,
) -> sqlite3.Connection:
    """Patch the memory server's connection factory to use the in-memory DB.

    Returns the connection so tests can assert directly against the stored
    rows after driving the tools.
    """
    monkeypatch.setattr("mcp_integration.memory_server._get_connection", lambda: memory_db)
    return memory_db
