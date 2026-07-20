"""Unit tests for `agents.personal_assistant.PersonalAssistant`.

The Anthropic client is fully mocked; these tests do not perform any real
network I/O and can run in any CI environment without secrets.
"""

from __future__ import annotations

from datetime import datetime, tzinfo
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from agents.personal_assistant import PersonalAssistant, ToolSpec
from core.config import Settings
from tests.conftest import FROZEN_NOW
from tools.save_note import NoteSchema, SaveNoteResult


@pytest.fixture
def knowledge_dir(tmp_path: Path) -> Path:
    """Create a temporary knowledge directory with two markdown files."""
    (tmp_path / "one.md").write_text("# One\nFirst knowledge chunk.", encoding="utf-8")
    (tmp_path / "two.md").write_text("# Two\nSecond knowledge chunk.", encoding="utf-8")
    return tmp_path


@pytest.fixture
def settings(knowledge_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Build a `Settings` instance pointing at the temporary knowledge dir."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
    monkeypatch.setenv("KNOWLEDGE_DIR", str(knowledge_dir))
    return Settings(_env_file=None)


class TestKnowledgeLoading:
    """The assistant must embed every markdown file into its system prompt."""

    def test_system_prompt_contains_all_knowledge_files(self, settings: Settings) -> None:
        with patch("agents.personal_assistant.AsyncAnthropic"):
            assistant = PersonalAssistant(settings=settings)

        prompt = assistant._system_prompt
        assert "First knowledge chunk." in prompt
        assert "Second knowledge chunk." in prompt
        assert "one.md" in prompt
        assert "two.md" in prompt

    def test_missing_knowledge_dir_produces_preamble_only(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        missing = tmp_path / "does-not-exist"
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        monkeypatch.setenv("KNOWLEDGE_DIR", str(missing))
        settings = Settings(_env_file=None)

        with patch("agents.personal_assistant.AsyncAnthropic"):
            assistant = PersonalAssistant(settings=settings)

        assert "KNOWLEDGE BASE" not in assistant._system_prompt
        assert "PlanSmart founder" in assistant._system_prompt


class TestAsk:
    """The async `ask` method must call the SDK correctly and parse the reply."""

    async def test_ask_returns_concatenated_text_blocks(self, settings: Settings) -> None:
        response = SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="Hello "),
                SimpleNamespace(type="text", text="world!"),
            ],
            stop_reason="end_turn",
        )

        with patch("agents.personal_assistant.AsyncAnthropic") as client_cls:
            client_cls.return_value.messages.create = AsyncMock(return_value=response)
            assistant = PersonalAssistant(settings=settings, tools=())
            reply = await assistant.ask("Hi there")

        assert reply == "Hello world!"
        create = client_cls.return_value.messages.create
        create.assert_awaited_once()

        call_kwargs = create.await_args.kwargs
        assert call_kwargs["model"] == settings.anthropic_model
        assert call_kwargs["max_tokens"] == settings.anthropic_max_tokens
        assert call_kwargs["messages"] == [{"role": "user", "content": "Hi there"}]
        # The static prompt is sent verbatim with the dynamic temporal
        # anchor appended after it.
        assert call_kwargs["system"].startswith(assistant._system_prompt)
        assert "Current System Time:" in call_kwargs["system"]

    async def test_ask_ignores_non_text_content_blocks(self, settings: Settings) -> None:
        response = SimpleNamespace(
            content=[
                SimpleNamespace(type="tool_use", text="ignored"),
                SimpleNamespace(type="text", text="only this"),
            ],
            stop_reason="end_turn",
        )

        with patch("agents.personal_assistant.AsyncAnthropic") as client_cls:
            client_cls.return_value.messages.create = AsyncMock(return_value=response)
            assistant = PersonalAssistant(settings=settings, tools=())
            reply = await assistant.ask("prompt")

        assert reply == "only this"

    async def test_ask_uses_the_secret_api_key(self, settings: Settings) -> None:
        response = _text_response("ok")

        with patch("agents.personal_assistant.AsyncAnthropic") as client_cls:
            client_cls.return_value.messages.create = AsyncMock(return_value=response)
            PersonalAssistant(settings=settings, tools=())

        client_cls.assert_called_once_with(api_key="sk-test-key")


def _text_response(text: str) -> SimpleNamespace:
    """Minimal end-of-turn Anthropic response with a single text block."""
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason="end_turn",
    )


class TestTemporalAnchor:
    """Every API call carries a fresh `Current System Time` anchor."""

    async def test_sent_system_prompt_contains_the_frozen_timestamp(
        self, settings: Settings
    ) -> None:
        with patch("agents.personal_assistant.AsyncAnthropic") as client_cls:
            create = AsyncMock(return_value=_text_response("ok"))
            client_cls.return_value.messages.create = create
            assistant = PersonalAssistant(settings=settings, tools=())
            await assistant.ask("hi")

        system = create.await_args_list[0].kwargs["system"]
        expected = (
            f"Current System Time: {FROZEN_NOW.isoformat()}. "
            "Use this as your temporal anchor for all queries."
        )
        assert system.endswith(expected)

    def test_cached_static_prompt_carries_no_timestamp(self, settings: Settings) -> None:
        with patch("agents.personal_assistant.AsyncAnthropic"):
            assistant = PersonalAssistant(settings=settings, tools=())
        assert "Current System Time:" not in assistant._system_prompt

    async def test_anchor_is_recomputed_on_every_api_call(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two sequential asks must observe two different clock readings,
        proving the anchor is evaluated per call rather than captured at
        construction time."""
        ticks = iter(
            [
                datetime(2026, 7, 17, 8, 0, 0),
                datetime(2026, 7, 17, 9, 30, 0),
            ]
        )

        class _TickingClock:
            @staticmethod
            def now(tz: tzinfo | None = None) -> datetime:
                return next(ticks)

        monkeypatch.setattr("agents.personal_assistant.datetime", _TickingClock)

        with patch("agents.personal_assistant.AsyncAnthropic") as client_cls:
            create = AsyncMock(side_effect=[_text_response("r1"), _text_response("r2")])
            client_cls.return_value.messages.create = create
            assistant = PersonalAssistant(settings=settings, tools=())
            await assistant.ask("q1")
            await assistant.ask("q2")

        first = create.await_args_list[0].kwargs["system"]
        second = create.await_args_list[1].kwargs["system"]
        assert "Current System Time: 2026-07-17T08:00:00." in first
        assert "Current System Time: 2026-07-17T09:30:00." in second


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


class TestConversationHistory:
    """Multi-turn state is preserved across `ask()` calls."""

    async def test_first_ask_produces_user_and_assistant_pair(self, settings: Settings) -> None:
        with patch("agents.personal_assistant.AsyncAnthropic") as client_cls:
            client_cls.return_value.messages.create = AsyncMock(
                return_value=_text_response("hi back")
            )
            assistant = PersonalAssistant(settings=settings, tools=())
            await assistant.ask("hello")

        assert assistant.history == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi back"},
        ]

    async def test_second_ask_includes_previous_turn(self, settings: Settings) -> None:
        with patch("agents.personal_assistant.AsyncAnthropic") as client_cls:
            create = AsyncMock(side_effect=[_text_response("reply1"), _text_response("reply2")])
            client_cls.return_value.messages.create = create
            assistant = PersonalAssistant(settings=settings, tools=())
            await assistant.ask("q1")
            await assistant.ask("q2")

        second_call = create.await_args_list[1]
        assert second_call.kwargs["messages"] == [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "reply1"},
            {"role": "user", "content": "q2"},
        ]

    async def test_history_property_returns_defensive_copy(self, settings: Settings) -> None:
        with patch("agents.personal_assistant.AsyncAnthropic") as client_cls:
            client_cls.return_value.messages.create = AsyncMock(return_value=_text_response("ok"))
            assistant = PersonalAssistant(settings=settings, tools=())
            await assistant.ask("hi")

        snapshot = assistant.history
        snapshot.append({"role": "user", "content": "tampered"})
        assert len(assistant.history) == 2

    async def test_reset_clears_history(self, settings: Settings) -> None:
        with patch("agents.personal_assistant.AsyncAnthropic") as client_cls:
            client_cls.return_value.messages.create = AsyncMock(return_value=_text_response("ok"))
            assistant = PersonalAssistant(settings=settings, tools=())
            await assistant.ask("hi")
            assert len(assistant.history) == 2

            assistant.reset()
            assert assistant.history == []

    async def test_history_unchanged_when_api_call_raises(self, settings: Settings) -> None:
        with patch("agents.personal_assistant.AsyncAnthropic") as client_cls:
            client_cls.return_value.messages.create = AsyncMock(side_effect=RuntimeError("boom"))
            assistant = PersonalAssistant(settings=settings, tools=())
            with pytest.raises(RuntimeError):
                await assistant.ask("this will fail")

        assert assistant.history == []


class TestHistoryTrimming:
    """History is bounded by `max_history_turns` and never sent in an invalid shape."""

    async def test_history_is_trimmed_after_exceeding_limit(
        self,
        monkeypatch: pytest.MonkeyPatch,
        knowledge_dir: Path,
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        monkeypatch.setenv("KNOWLEDGE_DIR", str(knowledge_dir))
        monkeypatch.setenv("MAX_HISTORY_TURNS", "4")
        constrained = Settings(_env_file=None)

        replies = [_text_response(f"r{i}") for i in range(1, 4)]
        with patch("agents.personal_assistant.AsyncAnthropic") as client_cls:
            create = AsyncMock(side_effect=replies)
            client_cls.return_value.messages.create = create
            assistant = PersonalAssistant(settings=constrained, tools=())
            await assistant.ask("q1")
            await assistant.ask("q2")
            await assistant.ask("q3")

        assert len(assistant.history) <= constrained.max_history_turns
        assert assistant.history[0]["role"] == "user"

        sent = create.await_args_list[2].kwargs["messages"]
        assert sent[0]["role"] == "user"
        assert len(sent) <= constrained.max_history_turns


class TestToolUseLoop:
    """The bounded tool-use loop dispatches tools, recovers from errors,
    and preserves invariant conversation shape end-to-end."""

    async def test_successful_tool_call_returns_final_text(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        captured_note: dict[str, Any] = {}

        def _fake_save(note: NoteSchema) -> SaveNoteResult:
            captured_note.update(note.model_dump())
            return SaveNoteResult(
                path=tmp_path / "2026-07-04-standup.md",
                filename="2026-07-04-standup.md",
                bytes_written=42,
            )

        tool = ToolSpec(
            name="save_note",
            description="Persist a markdown note.",
            schema=NoteSchema,
            call=_fake_save,
        )

        with patch("agents.personal_assistant.AsyncAnthropic") as client_cls:
            create = AsyncMock(
                side_effect=[
                    _tool_use_response(
                        "tu_1",
                        "save_note",
                        {
                            "slug": "standup",
                            "content": "notes from standup",
                            "tags": [],
                        },
                    ),
                    _text_response("Saved as 2026-07-04-standup.md."),
                ]
            )
            client_cls.return_value.messages.create = create

            uses: list[tuple[str, dict[str, Any]]] = []
            results: list[tuple[str, str, bool]] = []
            assistant = PersonalAssistant(settings=settings, tools=[tool])
            reply = await assistant.ask(
                "please save this",
                on_tool_use=lambda name, payload: uses.append((name, payload)),
                on_tool_result=lambda name, output, is_error: results.append(
                    (name, output, is_error)
                ),
            )

        assert reply == "Saved as 2026-07-04-standup.md."
        assert captured_note["slug"] == "standup"
        assert uses == [
            (
                "save_note",
                {"slug": "standup", "content": "notes from standup", "tags": []},
            )
        ]
        assert len(results) == 1
        name, output, is_error = results[0]
        assert name == "save_note"
        assert is_error is False
        assert "2026-07-04-standup.md" in output

        second_call = create.await_args_list[1].kwargs["messages"]
        last_message = second_call[-1]
        assert last_message["role"] == "user"
        assert isinstance(last_message["content"], list)
        block = last_message["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "tu_1"
        assert block["is_error"] is False

    async def test_tool_error_is_surfaced_and_llm_can_recover(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        """A pydantic validation failure on the first tool call is fed back
        to the model as `is_error=True`; the model retries with valid
        input on the second turn and the loop completes normally."""
        executed_slugs: list[str] = []

        def _fake_save(note: NoteSchema) -> SaveNoteResult:
            executed_slugs.append(note.slug)
            return SaveNoteResult(
                path=tmp_path / f"2026-07-04-{note.slug}.md",
                filename=f"2026-07-04-{note.slug}.md",
                bytes_written=10,
            )

        tool = ToolSpec(
            name="save_note",
            description="Persist a markdown note.",
            schema=NoteSchema,
            call=_fake_save,
        )

        with patch("agents.personal_assistant.AsyncAnthropic") as client_cls:
            create = AsyncMock(
                side_effect=[
                    _tool_use_response(
                        "tu_bad",
                        "save_note",
                        {
                            "slug": "Invalid Slug!!!",
                            "content": "attempt",
                            "tags": [],
                        },
                    ),
                    _tool_use_response(
                        "tu_good",
                        "save_note",
                        {
                            "slug": "recovered",
                            "content": "attempt",
                            "tags": [],
                        },
                    ),
                    _text_response("Saved after correction."),
                ]
            )
            client_cls.return_value.messages.create = create

            results: list[tuple[str, str, bool]] = []
            assistant = PersonalAssistant(settings=settings, tools=[tool])
            reply = await assistant.ask(
                "save a bad note",
                on_tool_result=lambda name, output, is_error: results.append(
                    (name, output, is_error)
                ),
            )

        assert executed_slugs == ["recovered"]
        assert reply == "Saved after correction."

        assert results[0][2] is True
        assert "ValidationError" in results[0][1]
        assert results[1][2] is False

        second_call_messages = create.await_args_list[1].kwargs["messages"]
        error_carrier = second_call_messages[-1]
        assert error_carrier["role"] == "user"
        assert error_carrier["content"][0]["is_error"] is True
        assert "ValidationError" in error_carrier["content"][0]["content"]

    async def test_unknown_tool_is_returned_as_error(self, settings: Settings) -> None:
        with patch("agents.personal_assistant.AsyncAnthropic") as client_cls:
            create = AsyncMock(
                side_effect=[
                    _tool_use_response("tu_x", "does_not_exist", {}),
                    _text_response("Recovered from an unknown tool."),
                ]
            )
            client_cls.return_value.messages.create = create
            results: list[tuple[str, str, bool]] = []
            assistant = PersonalAssistant(settings=settings, tools=())
            reply = await assistant.ask(
                "call a bad tool",
                on_tool_result=lambda name, output, is_error: results.append(
                    (name, output, is_error)
                ),
            )

        assert reply == "Recovered from an unknown tool."
        assert results[0][2] is True
        assert "Unknown tool" in results[0][1]

    async def test_iteration_ceiling_returns_graceful_message(
        self,
        monkeypatch: pytest.MonkeyPatch,
        knowledge_dir: Path,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        monkeypatch.setenv("KNOWLEDGE_DIR", str(knowledge_dir))
        monkeypatch.setenv("MAX_TOOL_ITERATIONS", "2")
        constrained = Settings(_env_file=None)

        tool = ToolSpec(
            name="save_note",
            description="Persist a markdown note.",
            schema=NoteSchema,
            call=lambda note: SaveNoteResult(
                path=tmp_path / "x.md", filename="x.md", bytes_written=1
            ),
        )

        def infinite_tool_use(counter: int) -> SimpleNamespace:
            return _tool_use_response(
                f"tu_{counter}",
                "save_note",
                {"slug": "loop", "content": "again", "tags": []},
            )

        with patch("agents.personal_assistant.AsyncAnthropic") as client_cls:
            client_cls.return_value.messages.create = AsyncMock(
                side_effect=[infinite_tool_use(1), infinite_tool_use(2)]
            )
            assistant = PersonalAssistant(settings=constrained, tools=[tool])
            reply = await assistant.ask("loop forever")

        assert "iteration ceiling" in reply.lower()
        # After ceiling the tail must be a synthesized assistant text so
        # the next turn sees a well-formed shape.
        assert assistant.history[-1] == {"role": "assistant", "content": reply}

    async def test_history_after_tool_round_preserves_pair(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        tool = ToolSpec(
            name="save_note",
            description="Persist a markdown note.",
            schema=NoteSchema,
            call=lambda note: SaveNoteResult(
                path=tmp_path / "x.md", filename="x.md", bytes_written=1
            ),
        )
        with patch("agents.personal_assistant.AsyncAnthropic") as client_cls:
            client_cls.return_value.messages.create = AsyncMock(
                side_effect=[
                    _tool_use_response(
                        "tu_1",
                        "save_note",
                        {"slug": "note", "content": "body", "tags": []},
                    ),
                    _text_response("Done."),
                ]
            )
            assistant = PersonalAssistant(settings=settings, tools=[tool])
            await assistant.ask("save it")

        history = assistant.history
        assert history[0]["role"] == "user"

        assert history[1]["role"] == "assistant"
        assert isinstance(history[1]["content"], list)
        assert any(block.get("type") == "tool_use" for block in history[1]["content"])

        assert history[2]["role"] == "user"
        assert isinstance(history[2]["content"], list)
        assert any(block.get("type") == "tool_result" for block in history[2]["content"])

        assert history[3] == {"role": "assistant", "content": "Done."}

    async def test_tool_use_callback_exception_does_not_break_the_loop(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        tool = ToolSpec(
            name="save_note",
            description="Persist a markdown note.",
            schema=NoteSchema,
            call=lambda note: SaveNoteResult(
                path=tmp_path / "x.md", filename="x.md", bytes_written=1
            ),
        )

        def _explosive_callback(name: str, payload: dict[str, Any]) -> None:
            raise RuntimeError("callback boom")

        with patch("agents.personal_assistant.AsyncAnthropic") as client_cls:
            client_cls.return_value.messages.create = AsyncMock(
                side_effect=[
                    _tool_use_response(
                        "tu_1",
                        "save_note",
                        {"slug": "safe", "content": "x", "tags": []},
                    ),
                    _text_response("Handled."),
                ]
            )
            assistant = PersonalAssistant(settings=settings, tools=[tool])
            reply = await assistant.ask(
                "save it",
                on_tool_use=_explosive_callback,
            )

        assert reply == "Handled."
