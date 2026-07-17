"""Model Context Protocol (MCP) transport for the PlanSmart assistant.

This package is the seam between a **Tool** (a Tavily web search exposed
over a stdio MCP server) and the **Agent** (which discovers and invokes
that tool through an MCP client). It is deliberately split so that the
Agent layer never has to import the MCP or Tavily SDKs:

* ``types``        — SDK-free value types and the ``MCPToolProvider``
                     protocol the Agent depends on.
* ``tavily_server`` — a ``FastMCP`` server, run as a subprocess, that
                     exposes the single ``tavily_search`` tool.
* ``client``       — ``TavilyMCPClient``, which spawns the server over
                     stdio and satisfies ``MCPToolProvider``.

The package is named ``mcp_integration`` rather than ``mcp`` on purpose:
a top-level ``mcp`` package would shadow the official ``mcp`` SDK on the
import path (``pythonpath = ["."]``) and break every SDK import.
"""

from __future__ import annotations

from mcp_integration.types import MCPToolError, MCPToolInfo, MCPToolProvider

__all__ = [
    "MCPToolError",
    "MCPToolInfo",
    "MCPToolProvider",
]
