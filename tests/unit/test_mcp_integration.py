"""Integration tests for MCP tools inside `PersonalAssistant`.

The Anthropic client is fully mocked and the MCP provider is the in-memory
`FakeMCPProvider` from `tests/conftest.py`, so these tests prove the WAT
wiring — discover tools via `list_tools()`, publish them to the model, and
route a selected call through `call_tool()` — with zero network or
subprocess involvement.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from agents.personal_assistant import MCPTool, PersonalAssistant, ToolSpec
from core.config import Settings
from tests.conftest import FakeMCPProvider, make_tool_info


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """A `Settings` instance pointed at an empty temporary knowledge dir."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
    monkeypatch.setenv("KNOWLEDGE_DIR", str(tmp_path))
    return Settings(_env_file=None)


def _text_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason="end_turn",
    )


def _tool_use_response(tool_id: str, name: str, tool_input: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", id=tool_id, name=name, input=tool_input)],
        stop_reason="tool_use",
    )


class TestMCPToolAdapter:
    """`MCPTool` bridges the AgentTool protocol to the MCP provider."""

    def test_input_schema_returns_server_json_schema(self) -> None:
        tool = MCPTool(
            name="tavily_search",
            description="d",
            json_schema={"type": "object", "properties": {"query": {}}},
            provider=FakeMCPProvider(),
        )
        assert tool.input_schema() == {"type": "object", "properties": {"query": {}}}

    async def test_run_routes_arguments_to_provider(self) -> None:
        provider = FakeMCPProvider(responses={"tavily_search": "routed output"})
        tool = MCPTool(
            name="tavily_search",
            description="d",
            json_schema={"type": "object"},
            provider=provider,
        )
        out = await tool.run({"query": "x", "max_results": 2})
        assert out == "routed output"
        assert provider.calls == [("tavily_search", {"query": "x", "max_results": 2})]


class TestRegisterMCPTools:
    """`register_mcp_tools` merges MCP tools into the registry."""

    async def test_registered_tool_appears_in_definitions(self, settings: Settings) -> None:
        with patch("agents.personal_assistant.AsyncAnthropic"):
            assistant = PersonalAssistant(settings=settings, tools=())
            await assistant.register_mcp_tools(FakeMCPProvider())

        definitions = assistant._build_tool_definitions()
        by_name = {d["name"]: d for d in definitions}
        assert "tavily_search" in by_name
        assert "query" in by_name["tavily_search"]["input_schema"]["properties"]

    async def test_registration_rebuilds_the_tool_manifest(self, settings: Settings) -> None:
        with patch("agents.personal_assistant.AsyncAnthropic"):
            assistant = PersonalAssistant(settings=settings, tools=())
            assert "tavily_search" not in assistant._system_prompt
            await assistant.register_mcp_tools(FakeMCPProvider())

        assert "tavily_search" in assistant._system_prompt

    async def test_name_collision_keeps_the_local_tool(self, settings: Settings) -> None:
        # Default registry already contains a local `web_search` ToolSpec.
        provider = FakeMCPProvider(tools=[make_tool_info(name="web_search")])
        with patch("agents.personal_assistant.AsyncAnthropic"):
            assistant = PersonalAssistant(settings=settings)
            await assistant.register_mcp_tools(provider)

        assert isinstance(assistant._tools["web_search"], ToolSpec)


class TestAskRoutesThroughMCP:
    """End-to-end: the model selects an MCP tool and gets its output back."""

    async def test_selected_mcp_tool_is_executed_via_provider(self, settings: Settings) -> None:
        provider = FakeMCPProvider(responses={"tavily_search": "Search results for: news"})
        with patch("agents.personal_assistant.AsyncAnthropic") as client_cls:
            create = AsyncMock(
                side_effect=[
                    _tool_use_response(
                        "tu_1", "tavily_search", {"query": "news", "max_results": 3}
                    ),
                    _text_response("Here is what I found."),
                ]
            )
            client_cls.return_value.messages.create = create

            results: list[tuple[str, str, bool]] = []
            assistant = PersonalAssistant(settings=settings, tools=())
            await assistant.register_mcp_tools(provider)
            reply = await assistant.ask(
                "search the news",
                on_tool_result=lambda name, output, is_error: results.append(
                    (name, output, is_error)
                ),
            )

        assert reply == "Here is what I found."
        assert provider.calls == [("tavily_search", {"query": "news", "max_results": 3})]
        assert results == [("tavily_search", "Search results for: news", False)]

        tool_result_message = create.await_args_list[1].kwargs["messages"][-1]
        block = tool_result_message["content"][0]
        assert block["type"] == "tool_result"
        assert block["content"] == "Search results for: news"
        assert block["is_error"] is False

    async def test_mcp_tool_error_is_surfaced_and_recoverable(self, settings: Settings) -> None:
        provider = FakeMCPProvider(errors=["tavily_search"])
        with patch("agents.personal_assistant.AsyncAnthropic") as client_cls:
            create = AsyncMock(
                side_effect=[
                    _tool_use_response("tu_bad", "tavily_search", {"query": "x"}),
                    _text_response("Recovered from the MCP failure."),
                ]
            )
            client_cls.return_value.messages.create = create

            results: list[tuple[str, str, bool]] = []
            assistant = PersonalAssistant(settings=settings, tools=())
            await assistant.register_mcp_tools(provider)
            reply = await assistant.ask(
                "search",
                on_tool_result=lambda name, output, is_error: results.append(
                    (name, output, is_error)
                ),
            )

        assert reply == "Recovered from the MCP failure."
        assert results[0][2] is True
        assert "simulated" in results[0][1]
        assert "MCPToolError" in results[0][1]


class TestMultipleMCPProviders:
    """Tools from two servers (e.g. Tavily + memory) coexist and route independently."""

    async def test_two_providers_register_and_dispatch_to_their_own_server(
        self, settings: Settings
    ) -> None:
        web = FakeMCPProvider(
            tools=[make_tool_info(name="tavily_search")],
            responses={"tavily_search": "web hit"},
        )
        memory = FakeMCPProvider(
            tools=[
                make_tool_info(name="store_fact"),
                make_tool_info(name="retrieve_context"),
            ],
            responses={"store_fact": "stored", "retrieve_context": "recalled"},
        )

        with patch("agents.personal_assistant.AsyncAnthropic"):
            assistant = PersonalAssistant(settings=settings, tools=())
            await assistant.register_mcp_tools(web)
            await assistant.register_mcp_tools(memory)

        names = {d["name"] for d in assistant._build_tool_definitions()}
        assert {"tavily_search", "store_fact", "retrieve_context"} <= names

        # Each MCPTool carries its own provider, so calls route to the
        # server that advertised the tool — no shared routing table needed.
        assert await assistant._tools["tavily_search"].run({"query": "x"}) == "web hit"
        assert await assistant._tools["store_fact"].run({"entity": "e", "fact": "f"}) == "stored"
        assert await assistant._tools["retrieve_context"].run({"query": "e"}) == "recalled"

        assert web.calls == [("tavily_search", {"query": "x"})]
        assert memory.calls == [
            ("store_fact", {"entity": "e", "fact": "f"}),
            ("retrieve_context", {"query": "e"}),
        ]
