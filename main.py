"""Interactive entry point for the PlanSmart personal assistant.

Run with:
    python main.py

The REPL prints three kinds of lines:

    you >         Your next prompt (input echo).
    [system]      Session lifecycle notifications and errors.
    [agent]       Live tool-use narration — one line when the model
                  requests a tool, one line when it returns or fails.
    agent >       The model's final natural-language reply for the turn.

Empty input or Ctrl-C / Ctrl-D exits. Type `/reset` (alias: `/clear`) to
discard the accumulated conversation history without restarting.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any, Final

from agents.personal_assistant import PersonalAssistant
from core.config import get_settings
from utils.logger import configure_logging, get_logger

_RESET_COMMANDS: Final[frozenset[str]] = frozenset({"/reset", "/clear"})
_PROMPT: Final[str] = "you > "
_SHORT_LIMIT: Final[int] = 80


def _emit(prefix: str, text: str) -> None:
    """Write a chat line straight to stdout, bypassing the structured logger."""
    sys.stdout.write(f"{prefix} {text}\n")
    sys.stdout.flush()


def _shorten(value: Any, limit: int = _SHORT_LIMIT) -> str:
    text = value if isinstance(value, str) else repr(value)
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _on_tool_use(name: str, tool_input: dict[str, Any]) -> None:
    """Announce that the model has requested a tool invocation."""
    if tool_input:
        rendered = ", ".join(
            f"{key}={_shorten(value, 40)}" for key, value in tool_input.items()
        )
        _emit("[agent]", f"calling tool: {name}({rendered})")
    else:
        _emit("[agent]", f"calling tool: {name}()")


def _on_tool_result(name: str, output: str, is_error: bool) -> None:
    """Report the outcome of a tool invocation on a single line."""
    tag = "error" if is_error else "ok"
    _emit("[agent]", f"tool {name} {tag}: {_shorten(output)}")


async def _run() -> None:
    """Async REPL that forwards user input to the assistant."""
    settings = get_settings()
    # In interactive mode we keep the terminal clean by suppressing the
    # structured logger below WARNING. Chat narration is emitted directly
    # via `_emit`, which never routes through the logger.
    configure_logging(level="WARNING")
    logger = get_logger("main")

    assistant = PersonalAssistant(settings=settings)

    _emit(
        "[system]",
        "PlanSmart assistant online. /reset to clear history, empty line to quit.",
    )
    loop = asyncio.get_running_loop()

    while True:
        try:
            user_input = await loop.run_in_executor(None, input, _PROMPT)
        except (EOFError, KeyboardInterrupt):
            _emit("[system]", "Session terminated by user.")
            return

        prompt = user_input.strip()
        if not prompt:
            _emit("[system]", "Empty input received. Exiting.")
            return

        if prompt in _RESET_COMMANDS:
            assistant.reset()
            _emit("[system]", "Conversation history cleared.")
            continue

        try:
            reply = await assistant.ask(
                prompt,
                on_tool_use=_on_tool_use,
                on_tool_result=_on_tool_result,
            )
        except Exception:
            logger.exception("Assistant call failed.")
            _emit(
                "[system]",
                "The assistant call failed. See stderr for the traceback.",
            )
            continue

        _emit("agent >", reply)


def main() -> None:
    """Synchronous wrapper around the async event loop."""
    asyncio.run(_run())


if __name__ == "__main__":
    main()
