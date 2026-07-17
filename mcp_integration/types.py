"""SDK-free value types shared across the MCP transport boundary.

WAT layer: **Tool** — pure data and a structural protocol, no I/O and no
reasoning. This module must never import the ``mcp`` or ``tavily`` SDKs:
the Agent layer imports ``MCPToolProvider`` from here to stay decoupled
from the transport, so importing the assistant must not drag in (or
require the installation of) either SDK.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


class MCPToolError(RuntimeError):
    """Raised when an MCP tool invocation reports an error result.

    The Agent layer catches this like any other tool failure and feeds
    the message back to the model as an ``is_error`` ``tool_result`` so
    the loop can self-correct rather than crash.
    """


@dataclass(frozen=True)
class MCPToolInfo:
    """A transport-agnostic description of a tool advertised by a server.

    Mirrors the fields the Agent layer needs to publish the tool to the
    Anthropic API, without leaking any ``mcp.types.Tool`` internals.
    """

    name: str
    description: str
    input_schema: dict[str, Any]


@runtime_checkable
class MCPToolProvider(Protocol):
    """The narrow surface the Agent layer needs from an MCP client.

    Any object that can enumerate tools and dispatch a call by name and
    argument mapping satisfies this protocol. Keeping it minimal lets
    tests substitute a trivial in-memory fake with no subprocess and no
    network — the offline-CI contract depends on it.
    """

    async def list_tools(self) -> Sequence[MCPToolInfo]:
        """Return the tools currently advertised by the connected server."""
        ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Invoke ``name`` with ``arguments`` and return its text output.

        Implementations must raise :class:`MCPToolError` when the server
        signals a tool-level error, so the Agent layer can surface it.
        """
        ...
