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
import logging
import sys
from contextlib import AsyncExitStack
from typing import Any, Final

from agents.personal_assistant import PersonalAssistant
from core.config import Settings, get_settings
from mcp_integration.client import TavilyMCPClient
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
        rendered = ", ".join(f"{key}={_shorten(value, 40)}" for key, value in tool_input.items())
        _emit("[agent]", f"calling tool: {name}({rendered})")
    else:
        _emit("[agent]", f"calling tool: {name}()")


def _on_tool_result(name: str, output: str, is_error: bool) -> None:
    """Report the outcome of a tool invocation on a single line."""
    tag = "error" if is_error else "ok"
    _emit("[agent]", f"tool {name} {tag}: {_shorten(output)}")


async def _connect_mcp(
    stack: AsyncExitStack,
    assistant: PersonalAssistant,
    settings: Settings,
    logger: logging.Logger,
) -> None:
    """Best-effort MCP startup: register Tavily tools or degrade gracefully.

    Failure to spawn or initialize the MCP server must never stop the
    assistant from running with its local tools, so any exception is
    logged and reported here rather than propagated. The client is
    entered into ``stack`` so its subprocess is torn down when the REPL
    exits.
    """
    try:
        client = await stack.enter_async_context(TavilyMCPClient(settings))
        await assistant.register_mcp_tools(client)
    except Exception:
        logger.exception("Tavily MCP startup failed; continuing with local tools only.")
        _emit("[system]", "Tavily MCP unavailable; continuing with local tools only.")
    else:
        _emit("[system]", "Tavily MCP server connected.")


async def _repl(assistant: PersonalAssistant, logger: logging.Logger) -> None:
    """Read-eval-print loop that forwards user input to the assistant."""
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


async def _run() -> None:
    """Wire up the assistant (with MCP tools) and run the REPL."""
    settings = get_settings()
    # In interactive mode we keep the terminal clean by suppressing the
    # structured logger below WARNING. Chat narration is emitted directly
    # via `_emit`, which never routes through the logger.
    configure_logging(level="WARNING")
    logger = get_logger("main")

    async with AsyncExitStack() as stack:
        assistant = PersonalAssistant(settings=settings)
        await _connect_mcp(stack, assistant, settings, logger)

        _emit(
            "[system]",
            "PlanSmart assistant online. /reset to clear history, empty line to quit.",
        )
        await _repl(assistant, logger)


def main() -> None:
    """Synchronous wrapper around the async event loop."""
    asyncio.run(_run())


if __name__ == "__main__":
    main()
