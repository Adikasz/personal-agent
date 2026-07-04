"""Trim conversation history without splitting tool_use / tool_result pairs.

WAT layer: **Tool** — pure computation, no LLM calls, no I/O, no global
state. The Anthropic Messages API imposes two structural invariants on
every message list it receives:

1. The conversation must begin with a message whose `role` is `user`.
2. Every content block of type `tool_use` produced by the assistant must
   be immediately followed by a user message whose content carries the
   matching `tool_result` block(s). Splitting the pair — for example by
   trimming the `tool_use` message but keeping the `tool_result` message,
   or by dropping the `tool_result` while retaining the `tool_use` — is a
   fatal 400 from the API.

`trim_history` is the single, tested chokepoint that enforces those
invariants when history is shortened. Every caller that needs to bound
history length must route through this function; no inline slicing.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from anthropic.types import MessageParam

from utils.logger import get_logger

__all__ = ["trim_history"]

_logger = get_logger(__name__)


@dataclass(frozen=True)
class _Atom:
    """An indivisible slice of conversation history.

    Either a single standalone message (a user question or a plain
    assistant answer) or a two-message tool round
    `(assistant_with_tool_use, user_with_tool_result)` that must be kept
    or dropped as one unit.
    """

    messages: tuple[MessageParam, ...]

    def __len__(self) -> int:
        return len(self.messages)

    @property
    def leading_role(self) -> str:
        return self.messages[0]["role"]


def _has_block_of_type(message: MessageParam, block_type: str) -> bool:
    """Return True if `message.content` is a block list containing `block_type`."""
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(
        isinstance(block, dict) and block.get("type") == block_type
        for block in content
    )


def _iter_atoms(messages: Sequence[MessageParam]) -> Iterator[_Atom]:
    """Group `messages` into ordered, indivisible atoms."""
    index = 0
    length = len(messages)
    while index < length:
        current = messages[index]
        is_tool_call = (
            current.get("role") == "assistant"
            and _has_block_of_type(current, "tool_use")
        )
        if is_tool_call and index + 1 < length:
            follower = messages[index + 1]
            if (
                follower.get("role") == "user"
                and _has_block_of_type(follower, "tool_result")
            ):
                yield _Atom((current, follower))
                index += 2
                continue

        if is_tool_call:
            _logger.warning(
                "tool_use at index %d has no matching tool_result; the atom "
                "will be treated as unpaired and may be dropped by the trimmer.",
                index,
            )

        yield _Atom((current,))
        index += 1


def trim_history(
    messages: Sequence[MessageParam],
    max_messages: int,
) -> list[MessageParam]:
    """Return `messages` trimmed from the head to satisfy the API invariants.

    Guarantees:
      1. `len(result) <= max_messages`.
      2. Every `tool_use` / `tool_result` pair present in the input is
         either fully retained or fully removed — never split.
      3. `result` is empty or starts with a `role: "user"` message.
      4. The relative order of retained messages matches the input.

    The input list is never mutated; a new list is returned. The leading
    -user invariant is enforced even for inputs already under the limit,
    so callers may hand this function raw slices without pre-validating.

    Args:
        messages: The full conversation history, in chronological order.
        max_messages: Upper bound on the number of retained messages.
            Non-positive values yield an empty result.

    Returns:
        A new, invariant-respecting slice of `messages`.
    """
    if max_messages <= 0:
        return []

    atoms = list(_iter_atoms(messages))
    remaining = sum(len(atom) for atom in atoms)

    head = 0
    while head < len(atoms):
        if remaining <= max_messages and atoms[head].leading_role == "user":
            break
        remaining -= len(atoms[head])
        head += 1

    return [message for atom in atoms[head:] for message in atom.messages]
