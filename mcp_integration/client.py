"""Stdio MCP client that spawns and drives the Tavily server subprocess.

WAT layer: **Tool** — deterministic transport plumbing with no LLM
reasoning. ``TavilyMCPClient`` launches ``mcp_integration.tavily_server``
as a child process over stdio, initializes an MCP session, and exposes
the two operations the Agent layer needs: enumerate tools and invoke one
by name. It satisfies :class:`mcp_integration.types.MCPToolProvider`.

Lifecycle is managed with a single :class:`contextlib.AsyncExitStack`,
so ``connect()``/``aclose()`` (and ``async with``) tear down the session
and the subprocess deterministically even if startup fails partway.
"""

from __future__ import annotations

import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from types import TracebackType
from typing import Any, Final

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import get_default_environment, stdio_client

from core.config import Settings, get_settings
from mcp_integration.types import MCPToolError, MCPToolInfo
from utils.logger import get_logger

__all__ = ["TavilyMCPClient"]

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
_SERVER_MODULE: Final[str] = "mcp_integration.tavily_server"


class TavilyMCPClient:
    """Manage a stdio MCP connection to the local Tavily server.

    The client is single-session: it spawns one subprocess on
    ``connect()`` and closes it on ``aclose()``. Reconnecting requires a
    fresh instance. Instances are async context managers, which is the
    recommended usage::

        async with TavilyMCPClient(settings) as client:
            tools = await client.list_tools()
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._logger = get_logger(self.__class__.__name__)
        self._exit_stack = AsyncExitStack()
        self._session: ClientSession | None = None

    @property
    def is_connected(self) -> bool:
        """Return whether an initialized MCP session is available."""
        return self._session is not None

    def _server_parameters(self) -> StdioServerParameters:
        """Describe how to launch the Tavily server subprocess.

        The child inherits a minimal, OS-appropriate environment plus the
        Tavily key and a ``PYTHONPATH`` entry for the project root so
        ``python -m mcp_integration.tavily_server`` resolves regardless of
        the parent's working directory.
        """
        env = get_default_environment()
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, [str(_PROJECT_ROOT), env.get("PYTHONPATH", "")])
        )
        env["TAVILY_API_KEY"] = self._settings.tavily_api_key.get_secret_value()
        return StdioServerParameters(
            command=sys.executable,
            args=["-m", _SERVER_MODULE],
            env=env,
            cwd=str(_PROJECT_ROOT),
        )

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
        self._logger.info("Connected to the Tavily MCP server over stdio.")

    async def aclose(self) -> None:
        """Close the MCP session and terminate the server subprocess."""
        await self._exit_stack.aclose()
        self._exit_stack = AsyncExitStack()
        self._session = None
        self._logger.info("Tavily MCP client closed.")

    async def __aenter__(self) -> TavilyMCPClient:
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
            raise MCPToolError("Tavily MCP client is not connected; call connect() first.")
        return self._session

    async def list_tools(self) -> list[MCPToolInfo]:
        """Return the tools advertised by the connected Tavily server."""
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
        self._logger.debug("MCP server advertised %d tool(s).", len(infos))
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


def _extract_text(content: Any) -> str:
    """Concatenate the text of every text block in an MCP tool result."""
    parts: list[str] = []
    for block in content or []:
        if getattr(block, "type", None) == "text":
            parts.append(str(getattr(block, "text", "")))
    return "\n".join(part for part in parts if part)
