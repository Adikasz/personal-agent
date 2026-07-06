"""Unit tests for `utils.history_manager.trim_history`.

These tests are the enforcement mechanism for the two Anthropic API
invariants documented in `utils/history_manager.py`. Every scenario
where a naive slice would split a `tool_use` / `tool_result` pair is
exercised here so that a regression fails fast, in isolation, without
any network I/O.

The module under test is pure computation; no fixtures beyond the shape
factories declared at the top of this file are required.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from anthropic.types import MessageParam

from utils.history_manager import trim_history as _run_trim


def _trim(messages: list[dict[str, Any]], max_messages: int) -> list[dict[str, Any]]:
    """Typed bridge around ``trim_history``.

    The factories below emit plain, message-shaped dicts on purpose: the
    trimmer is exercised structurally, including deliberately malformed
    shapes. The production function speaks the strict ``MessageParam``
    TypedDict, so the cast is localized to this single chokepoint instead
    of being scattered across every call site. No behavior changes.
    """
    result = _run_trim(cast("Sequence[MessageParam]", messages), max_messages)
    return cast("list[dict[str, Any]]", result)


# ---------------------------------------------------------------------------
# Message factories that mirror the Anthropic Messages API shape.
# ---------------------------------------------------------------------------


def _user(text: str) -> dict[str, Any]:
    """A plain user turn with string content."""
    return {"role": "user", "content": text}


def _assistant_text(text: str) -> dict[str, Any]:
    """A plain assistant turn with string content."""
    return {"role": "assistant", "content": text}


def _assistant_tool_use(
    tool_use_id: str,
    tool_name: str = "save_note",
    tool_input: dict[str, Any] | None = None,
    prefix_text: str | None = None,
) -> dict[str, Any]:
    """An assistant turn containing one `tool_use` block, optionally
    preceded by a text block (the LLM narrating before it calls the tool)."""
    blocks: list[dict[str, Any]] = []
    if prefix_text is not None:
        blocks.append({"type": "text", "text": prefix_text})
    blocks.append(
        {
            "type": "tool_use",
            "id": tool_use_id,
            "name": tool_name,
            "input": tool_input or {},
        }
    )
    return {"role": "assistant", "content": blocks}


def _user_tool_result(tool_use_id: str, output: str = "ok") -> dict[str, Any]:
    """A user turn carrying a matching `tool_result` block."""
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": output,
            }
        ],
    }


# ---------------------------------------------------------------------------
# Shared invariant assertions.
# ---------------------------------------------------------------------------


def _assert_no_split_pair(result: list[dict[str, Any]]) -> None:
    """Every retained `tool_use` id must be matched by a `tool_result` id
    in the immediately following user message."""
    for index, message in enumerate(result):
        if message["role"] != "assistant":
            continue
        content = message["content"]
        if not isinstance(content, list):
            continue
        tool_use_ids = {
            block["id"]
            for block in content
            if isinstance(block, dict) and block.get("type") == "tool_use"
        }
        if not tool_use_ids:
            continue
        assert index + 1 < len(result), (
            f"assistant tool_use at index {index} has no following user message"
        )
        follower = result[index + 1]
        assert follower["role"] == "user", (
            f"tool_use at index {index} must be followed by a user tool_result "
            f"message; got role={follower['role']!r}"
        )
        follower_content = follower["content"]
        assert isinstance(follower_content, list), (
            "tool_result carrier must have block-list content"
        )
        result_ids = {
            block["tool_use_id"]
            for block in follower_content
            if isinstance(block, dict) and block.get("type") == "tool_result"
        }
        assert tool_use_ids <= result_ids, (
            f"tool_use ids {tool_use_ids} missing matching tool_result ids (found {result_ids})"
        )


def _assert_no_orphan_tool_result(result: list[dict[str, Any]]) -> None:
    """No user message may carry a `tool_result` unless the previous
    assistant message emits the matching `tool_use`."""
    for index, message in enumerate(result):
        if message["role"] != "user":
            continue
        content = message["content"]
        if not isinstance(content, list):
            continue
        result_ids = {
            block["tool_use_id"]
            for block in content
            if isinstance(block, dict) and block.get("type") == "tool_result"
        }
        if not result_ids:
            continue
        assert index > 0, (
            "leading user message must not carry a tool_result — the "
            "matching tool_use has already been trimmed away"
        )
        previous = result[index - 1]
        assert previous["role"] == "assistant", (
            "tool_result must be preceded by an assistant tool_use message"
        )
        assert isinstance(previous["content"], list)
        use_ids = {
            block["id"]
            for block in previous["content"]
            if isinstance(block, dict) and block.get("type") == "tool_use"
        }
        assert result_ids <= use_ids, (
            f"tool_result ids {result_ids} lack matching tool_use in the "
            f"preceding message (found {use_ids})"
        )


def _assert_all_invariants(result: list[dict[str, Any]], max_messages: int) -> None:
    assert len(result) <= max_messages
    if result:
        assert result[0]["role"] == "user"
    _assert_no_split_pair(result)
    _assert_no_orphan_tool_result(result)


# ---------------------------------------------------------------------------
# Basic bounds and immutability.
# ---------------------------------------------------------------------------


class TestBounds:
    def test_empty_input_returns_empty(self) -> None:
        assert _trim([], 5) == []

    def test_zero_max_returns_empty(self) -> None:
        assert _trim([_user("hi")], 0) == []

    def test_negative_max_returns_empty(self) -> None:
        assert _trim([_user("hi"), _assistant_text("hey")], -3) == []

    def test_under_limit_returns_unchanged_when_leading_is_user(self) -> None:
        messages = [_user("q"), _assistant_text("a")]
        assert _trim(messages, 5) == messages

    def test_at_limit_returns_unchanged_when_leading_is_user(self) -> None:
        messages = [_user("q"), _assistant_text("a")]
        assert _trim(messages, 2) == messages

    def test_result_length_never_exceeds_max(self) -> None:
        messages = [_user(f"q{index}") for index in range(10)]
        result = _trim(messages, 3)
        assert len(result) <= 3

    def test_input_list_is_not_mutated(self) -> None:
        messages = [
            _user("q1"),
            _assistant_text("a1"),
            _user("q2"),
            _assistant_text("a2"),
        ]
        snapshot = [dict(message) for message in messages]
        _trim(messages, 2)
        assert messages == snapshot


# ---------------------------------------------------------------------------
# Leading-user invariant enforcement.
# ---------------------------------------------------------------------------


class TestLeadingUserInvariant:
    def test_head_assistant_is_dropped_when_over_limit(self) -> None:
        messages = [
            _user("q1"),
            _assistant_text("a1"),
            _user("q2"),
            _assistant_text("a2"),
        ]
        result = _trim(messages, 3)
        assert result[0]["role"] == "user"
        _assert_all_invariants(result, 3)

    def test_result_is_empty_when_no_user_head_fits(self) -> None:
        messages = [_user("q"), _assistant_text("a"), _assistant_text("a2")]
        assert _trim(messages, 1) == []

    def test_all_assistant_input_is_fully_dropped(self) -> None:
        messages = [_assistant_text("a1"), _assistant_text("a2")]
        assert _trim(messages, 5) == []

    def test_leading_user_invariant_enforced_even_when_under_limit(self) -> None:
        messages = [_assistant_text("orphan"), _user("q"), _assistant_text("a")]
        result = _trim(messages, 10)
        assert result == [_user("q"), _assistant_text("a")]


# ---------------------------------------------------------------------------
# Tool-round pair preservation.
# ---------------------------------------------------------------------------


class TestPairPreservation:
    def test_tool_round_retained_when_budget_allows(self) -> None:
        messages = [
            _user("save this"),
            _assistant_tool_use("tu_1"),
            _user_tool_result("tu_1"),
            _assistant_text("saved."),
        ]
        result = _trim(messages, 10)
        assert result == messages
        _assert_all_invariants(result, 10)

    def test_tool_round_dropped_together_when_budget_would_split_it(self) -> None:
        messages = [
            _user("q0"),
            _assistant_tool_use("tu_1"),
            _user_tool_result("tu_1"),
            _assistant_text("done"),
        ]
        result = _trim(messages, 3)
        # A naive tail-3 slice would return
        # [assistant_tool_use, user_tool_result, assistant_text],
        # which violates leading-user AND the API pair rule. The trimmer
        # must drop the pair entirely rather than split it.
        _assert_all_invariants(result, 3)
        for message in result:
            content = message["content"]
            if isinstance(content, list):
                for block in content:
                    assert block.get("type") not in {"tool_use", "tool_result"}

    def test_parallel_tool_uses_stay_bundled_when_retained(self) -> None:
        multi_use = {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "tu_1", "name": "save_note", "input": {}},
                {"type": "tool_use", "id": "tu_2", "name": "save_note", "input": {}},
            ],
        }
        multi_result = {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": "ok1"},
                {"type": "tool_result", "tool_use_id": "tu_2", "content": "ok2"},
            ],
        }
        messages = [_user("q"), multi_use, multi_result, _assistant_text("done")]
        result = _trim(messages, 10)
        assert result == messages
        _assert_all_invariants(result, 10)

    def test_parallel_tool_uses_dropped_together_when_budget_forces(self) -> None:
        multi_use = {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "tu_1", "name": "save_note", "input": {}},
                {"type": "tool_use", "id": "tu_2", "name": "save_note", "input": {}},
            ],
        }
        multi_result = {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": "ok1"},
                {"type": "tool_result", "tool_use_id": "tu_2", "content": "ok2"},
            ],
        }
        messages = [_user("q"), multi_use, multi_result, _assistant_text("done")]
        result = _trim(messages, 3)
        _assert_all_invariants(result, 3)

    def test_assistant_message_with_text_and_tool_use_treated_as_pair(self) -> None:
        messages = [
            _user("q"),
            _assistant_tool_use("tu_1", prefix_text="Thinking..."),
            _user_tool_result("tu_1"),
            _assistant_text("done"),
        ]
        result = _trim(messages, 10)
        assert result == messages
        _assert_all_invariants(result, 10)

    def test_realistic_transcript_across_all_budgets(self) -> None:
        messages = [
            _user("q1"),
            _assistant_tool_use("tu_a"),
            _user_tool_result("tu_a"),
            _assistant_text("a1"),
            _user("q2"),
            _assistant_tool_use("tu_b"),
            _user_tool_result("tu_b"),
            _assistant_text("a2"),
        ]
        for cap in range(0, len(messages) + 3):
            result = _trim(messages, cap)
            _assert_all_invariants(result, cap)


# ---------------------------------------------------------------------------
# Order preservation.
# ---------------------------------------------------------------------------


class TestOrderPreservation:
    def test_relative_order_of_retained_messages_is_stable(self) -> None:
        messages = [
            _user(f"q{index}") if index % 2 == 0 else _assistant_text(f"a{index}")
            for index in range(6)
        ]
        result = _trim(messages, 4)
        indices = [messages.index(message) for message in result]
        assert indices == sorted(indices)


# ---------------------------------------------------------------------------
# Malformed input — must not crash.
# ---------------------------------------------------------------------------


class TestUnpairedInput:
    def test_lone_trailing_tool_use_is_not_crashy(self) -> None:
        # Agent is mid-flight, still waiting for the tool_result — a
        # valid intermediate state that the caller must be free to
        # inspect before dispatching the tool call.
        messages = [_user("q"), _assistant_tool_use("tu_1")]
        result = _trim(messages, 5)
        assert result == messages

    def test_unpaired_tool_use_at_head_is_dropped(self) -> None:
        messages = [
            _user("q1"),
            _assistant_text("a1"),
            _assistant_tool_use("tu_orphan"),
        ]
        result = _trim(messages, 2)
        # A dangling assistant tool_use with no matching tool_result
        # cannot be a valid conversation head; the trimmer must drop it.
        for message in result:
            assert (
                message["role"] == "user"
                or not isinstance(message["content"], list)
                or not any(
                    isinstance(block, dict) and block.get("type") == "tool_use"
                    for block in message["content"]
                )
            )
