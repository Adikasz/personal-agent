"""Stdio MCP clients that spawn and drive the local server subprocesses.

WAT layer: **Tool** — deterministic transport plumbing with no LLM
reasoning. :class:`StdioMCPClient` launches one MCP server as a child
process over stdio, initializes a session, and exposes the two operations
the Agent layer needs: enumerate tools and invoke one by name. It
satisfies :class:`mcp_integration.types.MCPToolProvider`.

Two concrete servers are provided:

* :class:`TavilyMCPClient`  — ``mcp_integration.tavily_server`` (web search).
* :class:`MemoryMCPClient`  — ``mcp_integration.memory_server`` (long-term memory).

Both can run simultaneously; ``main.py`` enters each into a shared
:class:`contextlib.AsyncExitStack` and registers the tools of both with
the assistant. Each client owns a single :class:`AsyncExitStack`, so
``connect()``/``aclose()`` (and ``async with``) tear down the session and
the subprocess deterministically even if startup fails partway.
"""

from __future__ import annotations

import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from types import TracebackType
from typing import Any, Final, Self

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import get_default_environment, stdio_client

from core.config import Settings, get_settings
from mcp_integration.types import MCPToolError, MCPToolInfo
from utils.logger import get_logger

__all__ = ["MemoryMCPClient", "StdioMCPClient", "TavilyMCPClient"]

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
_TAVILY_MODULE: Final[str] = "mcp_integration.tavily_server"
_MEMORY_MODULE: Final[str] = "mcp_integration.memory_server"


class StdioMCPClient:
    """Manage a stdio MCP connection to one local server subprocess.

    Single-session: it spawns one subprocess on ``connect()`` and closes
    it on ``aclose()``; reconnecting requires a fresh instance. Instances
    are async context managers, which is the recommended usage::

        async with TavilyMCPClient(settings) as client:
            tools = await client.list_tools()

    Subclasses implement :meth:`_server_parameters` to describe which
    server to launch and which environment it needs.
    """

    #: Human-readable server label used in log lines and error messages.
    _LABEL: str = "MCP"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._logger = get_logger(self.__class__.__name__)
        self._exit_stack = AsyncExitStack()
        self._session: ClientSession | None = None

    @property
    def is_connected(self) -> bool:
        """Return whether an initialized MCP session is available."""
        return self._session is not None

    def _base_env(self) -> dict[str, str]:
        """Return a minimal, OS-appropriate environment for the subprocess.

        A ``PYTHONPATH`` entry for the project root is added so
        ``python -m mcp_integration.<server>`` resolves regardless of the
        parent's working directory.
        """
        env = get_default_environment()
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, [str(_PROJECT_ROOT), env.get("PYTHONPATH", "")])
        )
        return env

    def _params_for(self, module: str, env: dict[str, str]) -> StdioServerParameters:
        """Build launch parameters for ``python -m <module>``."""
        return StdioServerParameters(
            command=sys.executable,
            args=["-m", module],
            env=env,
            cwd=str(_PROJECT_ROOT),
        )

    def _server_parameters(self) -> StdioServerParameters:
        """Describe how to launch this client's server subprocess."""
        raise NotImplementedError

    async def connect(self) -> None:
        """Spawn the server and initialize the MCP session.

        Idempotent: a second call on a live client is a no-op. If any
        step fails, the partially built exit stack is unwound so no
        subprocess is left dangling.
        """
        if self._session is not None:
            self._logger.debug("connect() called on an already-connected client; ignoring.")
            return
        try:
            read_stream, write_stream = await self._exit_stack.enter_async_context(
                stdio_client(self._server_parameters())
            )
            session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await session.initialize()
        except BaseException:
            await self._exit_stack.aclose()
            self._exit_stack = AsyncExitStack()
            self._session = None
            raise
        self._session = session
        self._logger.info("Connected to the %s MCP server over stdio.", self._LABEL)

    async def aclose(self) -> None:
        """Close the MCP session and terminate the server subprocess."""
        await self._exit_stack.aclose()
        self._exit_stack = AsyncExitStack()
        self._session = None
        self._logger.info("%s MCP client closed.", self._LABEL)

    async def __aenter__(self) -> Self:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise MCPToolError(f"{self._LABEL} MCP client is not connected; call connect() first.")
        return self._session

    async def list_tools(self) -> list[MCPToolInfo]:
        """Return the tools advertised by the connected server."""
        session = self._require_session()
        result = await session.list_tools()
        infos = [
            MCPToolInfo(
                name=tool.name,
                description=tool.description or "",
                input_schema=dict(tool.inputSchema or {}),
            )
            for tool in result.tools
        ]
        self._logger.debug("%s MCP server advertised %d tool(s).", self._LABEL, len(infos))
        return infos

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Invoke ``name`` on the server and return its text content.

        Raises:
            MCPToolError: if the client is not connected, or the server
                reports a tool-level error result.
        """
        session = self._require_session()
        result = await session.call_tool(name, arguments)
        text = _extract_text(result.content)
        if bool(getattr(result, "isError", False)):
            raise MCPToolError(text or f"MCP tool {name!r} reported an error.")
        return text


class TavilyMCPClient(StdioMCPClient):
    """Stdio client for the Tavily web-search MCP server."""

    _LABEL = "Tavily"

    def _server_parameters(self) -> StdioServerParameters:
        env = self._base_env()
        env["TAVILY_API_KEY"] = self._settings.tavily_api_key.get_secret_value()
        return self._params_for(_TAVILY_MODULE, env)


class MemoryMCPClient(StdioMCPClient):
    """Stdio client for the persistent-memory MCP server."""

    _LABEL = "Memory"

    def _server_parameters(self) -> StdioServerParameters:
        env = self._base_env()
        env["MEMORY_DB_PATH"] = str(self._settings.resolved_memory_db_path)
        return self._params_for(_MEMORY_MODULE, env)


def _extract_text(content: Any) -> str:
    """Concatenate the text of every text block in an MCP tool result."""
    parts: list[str] = []
    for block in content or []:
        if getattr(block, "type", None) == "text":
            parts.append(str(getattr(block, "text", "")))
    return "\n".join(part for part in parts if part)
