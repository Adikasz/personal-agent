"""Unit tests for `mcp_integration.client.TavilyMCPClient`.

No subprocess is ever spawned: the `mock_mcp_transport` fixture (in
`tests/conftest.py`) patches `stdio_client` and `ClientSession` inside the
client module, and every session method is an `AsyncMock`. This keeps the
suite fully offline while exercising connect/close, tool discovery, result
extraction, and error propagation.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mcp_integration.client import TavilyMCPClient
from mcp_integration.types import MCPToolError, MCPToolInfo
from tests.conftest import make_call_tool_result


class TestServerParameters:
    """Subprocess launch parameters are derived without any I/O."""

    def test_parameters_target_the_tavily_server_module(self) -> None:
        params = TavilyMCPClient()._server_parameters()
        assert params.command == sys.executable
        assert params.args == ["-m", "mcp_integration.tavily_server"]

    def test_api_key_is_injected_into_subprocess_env(self) -> None:
        params = TavilyMCPClient()._server_parameters()
        assert params.env is not None
        assert params.env["TAVILY_API_KEY"] == "tvly-test-key"
        assert "PYTHONPATH" in params.env


class TestConnectionLifecycle:
    """connect()/aclose() and the async-context-manager protocol."""

    async def test_connect_initializes_and_marks_connected(
        self, mock_mcp_transport: SimpleNamespace
    ) -> None:
        client = TavilyMCPClient()
        assert client.is_connected is False

        await client.connect()
        assert client.is_connected is True
        mock_mcp_transport.initialize.assert_awaited_once()

        await client.aclose()
        assert client.is_connected is False

    async def test_context_manager_connects_and_closes(
        self, mock_mcp_transport: SimpleNamespace
    ) -> None:
        async with TavilyMCPClient() as client:
            assert client.is_connected is True
        assert client.is_connected is False

    async def test_connect_is_idempotent(self, mock_mcp_transport: SimpleNamespace) -> None:
        client = TavilyMCPClient()
        await client.connect()
        await client.connect()
        mock_mcp_transport.initialize.assert_awaited_once()
        await client.aclose()

    async def test_failed_initialize_unwinds_and_reports_disconnected(
        self, mock_mcp_transport: SimpleNamespace
    ) -> None:
        mock_mcp_transport.initialize = AsyncMock(side_effect=RuntimeError("handshake failed"))
        client = TavilyMCPClient()

        with pytest.raises(RuntimeError, match="handshake failed"):
            await client.connect()

        assert client.is_connected is False
        # The stack was reset; a subsequent aclose is safe and a no-op.
        await client.aclose()


class TestListTools:
    """`list_tools` maps `mcp.types.Tool` shapes into `MCPToolInfo`."""

    async def test_tools_are_mapped_to_info(self, mock_mcp_transport: SimpleNamespace) -> None:
        mock_mcp_transport.list_tools = AsyncMock(
            return_value=SimpleNamespace(
                tools=[
                    SimpleNamespace(
                        name="tavily_search",
                        description="Search the web.",
                        inputSchema={"type": "object", "properties": {"query": {}}},
                    )
                ]
            )
        )
        async with TavilyMCPClient() as client:
            infos = await client.list_tools()

        assert infos == [
            MCPToolInfo(
                name="tavily_search",
                description="Search the web.",
                input_schema={"type": "object", "properties": {"query": {}}},
            )
        ]

    async def test_missing_description_becomes_empty_string(
        self, mock_mcp_transport: SimpleNamespace
    ) -> None:
        mock_mcp_transport.list_tools = AsyncMock(
            return_value=SimpleNamespace(
                tools=[SimpleNamespace(name="t", description=None, inputSchema={})]
            )
        )
        async with TavilyMCPClient() as client:
            infos = await client.list_tools()
        assert infos[0].description == ""

    async def test_list_tools_before_connect_raises(self) -> None:
        with pytest.raises(MCPToolError, match="not connected"):
            await TavilyMCPClient().list_tools()


class TestCallTool:
    """`call_tool` extracts text and honors the server's error flag."""

    async def test_text_content_is_concatenated(self, mock_mcp_transport: SimpleNamespace) -> None:
        mock_mcp_transport.call_tool = AsyncMock(
            return_value=make_call_tool_result("hello from tavily")
        )
        async with TavilyMCPClient() as client:
            out = await client.call_tool("tavily_search", {"query": "x"})

        assert out == "hello from tavily"
        mock_mcp_transport.call_tool.assert_awaited_once_with("tavily_search", {"query": "x"})

    async def test_error_result_raises_mcp_tool_error(
        self, mock_mcp_transport: SimpleNamespace
    ) -> None:
        mock_mcp_transport.call_tool = AsyncMock(
            return_value=make_call_tool_result("upstream exploded", is_error=True)
        )
        async with TavilyMCPClient() as client:
            with pytest.raises(MCPToolError, match="upstream exploded"):
                await client.call_tool("tavily_search", {"query": "x"})

    async def test_non_text_blocks_are_ignored(self, mock_mcp_transport: SimpleNamespace) -> None:
        mock_mcp_transport.call_tool = AsyncMock(
            return_value=SimpleNamespace(
                content=[
                    SimpleNamespace(type="image", data="..."),
                    SimpleNamespace(type="text", text="only text survives"),
                ],
                isError=False,
            )
        )
        async with TavilyMCPClient() as client:
            out = await client.call_tool("tavily_search", {"query": "x"})
        assert out == "only text survives"

    async def test_call_tool_before_connect_raises(self) -> None:
        with pytest.raises(MCPToolError, match="not connected"):
            await TavilyMCPClient().call_tool("tavily_search", {"query": "x"})
