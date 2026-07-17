"""Unit tests for `mcp_integration.types`.

These types are the SDK-free contract the Agent layer depends on, so the
tests assert their shape and the structural-protocol conformance that keeps
the transport swappable (real client vs. in-memory fake).
"""

from __future__ import annotations

import dataclasses

import pytest

from mcp_integration.client import TavilyMCPClient
from mcp_integration.types import MCPToolError, MCPToolInfo, MCPToolProvider
from tests.conftest import FakeMCPProvider


class TestMCPToolInfo:
    """`MCPToolInfo` is an immutable value object."""

    def test_fields_round_trip(self) -> None:
        info = MCPToolInfo(
            name="tavily_search",
            description="Search the web.",
            input_schema={"type": "object"},
        )
        assert info.name == "tavily_search"
        assert info.description == "Search the web."
        assert info.input_schema == {"type": "object"}

    def test_is_frozen(self) -> None:
        info = MCPToolInfo(name="t", description="d", input_schema={})
        with pytest.raises(dataclasses.FrozenInstanceError):
            info.name = "changed"  # type: ignore[misc]


class TestMCPToolError:
    """`MCPToolError` behaves like the runtime error it is."""

    def test_is_runtime_error_subclass(self) -> None:
        assert issubclass(MCPToolError, RuntimeError)
        with pytest.raises(RuntimeError):
            raise MCPToolError("boom")


class TestProviderConformance:
    """Both the fake and the real client satisfy the runtime protocol."""

    def test_fake_provider_is_a_provider(self) -> None:
        assert isinstance(FakeMCPProvider(), MCPToolProvider)

    def test_real_client_is_a_provider(self) -> None:
        assert isinstance(TavilyMCPClient(), MCPToolProvider)
