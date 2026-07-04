"""End-to-end WAT loop test: workflow discovery and execution.

The Anthropic client is mocked with a scripted sequence of tool_use /
text responses. Every tool in the loop is the real production
implementation, so this test proves that:

- The agent's tool registry dispatches by name correctly.
- `list_directory` reads the real `workflows/` directory and surfaces
  `morning_briefing.md` back to the mocked LLM.
- `read_file` returns the real SOP text with the expected heading.
- `search_notes` runs against an isolated `tmp_path` scratchpad (via
  monkeypatched `DEFAULT_NOTES_DIR`) and reports a well-formed empty
  result.
- The tool-use loop threads each tool_result back into the next
  Anthropic request under the correct `tool_use_id`.

The test uses only substring assertions on the serialized tool_result
payloads so it stays robust against JSON field ordering and OS-specific
path separators.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from agents.personal_assistant import PersonalAssistant
from core.config import Settings


def _tool_use_response(
    tool_use_id: str,
    tool_name: str,
    tool_input: dict[str, Any],
) -> SimpleNamespace:
    """Anthropic response that requests a single tool invocation."""
    return SimpleNamespace(
        content=[
            SimpleNamespace(
                type="tool_use",
                id=tool_use_id,
                name=tool_name,
                input=tool_input,
            )
        ],
        stop_reason="tool_use",
    )


def _text_response(text: str) -> SimpleNamespace:
    """Anthropic response that terminates the turn with a plain text reply."""
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason="end_turn",
    )


class TestMorningBriefingWorkflowLoop:
    """Prove the LLM-driven WAT sequence executes correctly against the
    real workflow SOP and deterministic tool implementations."""

    async def test_agent_discovers_reads_and_executes_briefing_workflow(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # Isolate every mutable input the test depends on so the
        # assertions are stable across environments.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        knowledge_dir = tmp_path / "knowledge"
        knowledge_dir.mkdir()
        monkeypatch.setenv("KNOWLEDGE_DIR", str(knowledge_dir))
        settings = Settings(_env_file=None)

        # Sandbox the notes scratchpad the search tool consults, so
        # this test is not contaminated by real notes on disk.
        empty_notes = tmp_path / "notes"
        empty_notes.mkdir()
        monkeypatch.setattr(
            "tools.search_notes.DEFAULT_NOTES_DIR", empty_notes
        )

        # Scripted LLM: discover workflows, read the briefing SOP,
        # search for prior priorities, then respond with a final text.
        script = [
            _tool_use_response(
                "tu_list",
                "list_directory",
                {"relative_path": "workflows"},
            ),
            _tool_use_response(
                "tu_read",
                "read_file",
                {"filepath": "workflows/morning_briefing.md"},
            ),
            _tool_use_response(
                "tu_search",
                "search_notes",
                {"query": "priorities"},
            ),
            _text_response(
                "No prior priorities on file — let's start with an intake question."
            ),
        ]

        with patch("agents.personal_assistant.AsyncAnthropic") as client_cls:
            create = AsyncMock(side_effect=script)
            client_cls.return_value.messages.create = create

            assistant = PersonalAssistant(settings=settings)
            reply = await assistant.ask("give me a morning briefing")

        assert create.await_count == 4
        assert reply.startswith("No prior priorities")

        # Turn 2 must carry the list_directory tool_result naming the SOP.
        list_result = _last_tool_result(create.await_args_list[1])
        assert list_result["tool_use_id"] == "tu_list"
        assert list_result["is_error"] is False
        assert "morning_briefing.md" in list_result["content"]

        # Turn 3 must carry the read_file tool_result with the SOP title.
        read_result = _last_tool_result(create.await_args_list[2])
        assert read_result["tool_use_id"] == "tu_read"
        assert read_result["is_error"] is False
        assert "Morning Briefing" in read_result["content"]

        # Turn 4 must carry the search_notes tool_result with an empty
        # match list — the scratchpad is intentionally empty.
        search_result = _last_tool_result(create.await_args_list[3])
        assert search_result["tool_use_id"] == "tu_search"
        assert search_result["is_error"] is False
        compact = search_result["content"].replace(" ", "")
        assert '"scanned":0' in compact
        assert '"matches":[]' in compact

    async def test_traversal_attempt_surfaces_error_and_agent_recovers(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """A hallucinated path-traversal in `read_file` must not crash the
        loop; it must come back to the LLM as an `is_error=True`
        tool_result the model can react to."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        knowledge_dir = tmp_path / "knowledge"
        knowledge_dir.mkdir()
        monkeypatch.setenv("KNOWLEDGE_DIR", str(knowledge_dir))
        settings = Settings(_env_file=None)

        script = [
            _tool_use_response(
                "tu_attack",
                "read_file",
                {"filepath": "../../../etc/passwd"},
            ),
            _text_response("Understood; that path is out of bounds."),
        ]

        with patch("agents.personal_assistant.AsyncAnthropic") as client_cls:
            create = AsyncMock(side_effect=script)
            client_cls.return_value.messages.create = create

            assistant = PersonalAssistant(settings=settings)
            reply = await assistant.ask("read the passwd file")

        assert reply == "Understood; that path is out of bounds."
        error_result = _last_tool_result(create.await_args_list[1])
        assert error_result["tool_use_id"] == "tu_attack"
        assert error_result["is_error"] is True
        assert "SecurityError" in error_result["content"]


def _last_tool_result(call_record: Any) -> dict[str, Any]:
    """Return the trailing tool_result block from a mock's kwargs snapshot."""
    messages = call_record.kwargs["messages"]
    last = messages[-1]
    assert last["role"] == "user", (
        "The turn following a tool_use must be a user message carrying "
        f"tool_result blocks; got {last['role']!r}"
    )
    assert isinstance(last["content"], list), (
        "tool_result carrier must have block-list content"
    )
    block = last["content"][0]
    assert block["type"] == "tool_result"
    return block
