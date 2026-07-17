"""Personal assistant agent for the PlanSmart founder.

WAT layer: **Agent** — LLM-backed reasoning grounded in the local
`knowledge/` directory. Composes deterministic Tools (config, logger,
Anthropic API, and the tool registry) but implements no business logic
of its own.

The assistant runs a bounded tool-use loop on every `ask()` call:

1. Send the conversation to the model along with the tool schemas.
2. If the model requests one or more tool invocations, validate each
   tool input against its pydantic schema and execute the tool.
3. Errors raised during validation or execution are **never** propagated
   to the caller — they are formatted and fed back to the model inside
   the corresponding `tool_result` block with `is_error=True`, allowing
   the model to reason about the failure and retry autonomously.
4. Steps 1-3 repeat until the model returns a final answer or the
   configured iteration ceiling is reached.

Conversation history is bounded by `Settings.max_history_turns` and
trimmed through `utils.history_manager.trim_history`, which guarantees
that no `tool_use`/`tool_result` pair is ever split across a trim
boundary.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol, cast

from anthropic import AsyncAnthropic
from anthropic.types import ContentBlockParam, MessageParam, ToolResultBlockParam
from pydantic import BaseModel

from core.config import Settings, get_settings
from mcp_integration.types import MCPToolProvider
from tools.index_document import IndexDocumentQuery, index_document
from tools.list_directory import ListDirectoryQuery, list_directory
from tools.read_file import ReadFileQuery, read_file
from tools.save_note import NoteSchema, save_note
from tools.scrape_url import ScrapeUrlQuery, scrape_url
from tools.search_notes import SearchNotesQuery, search_notes
from tools.semantic_search import SemanticSearchQuery, semantic_search
from tools.web_search import WebSearchQuery, web_search
from utils.history_manager import trim_history
from utils.logger import get_logger

_KNOWLEDGE_GLOB: Final[str] = "*.md"
_MAX_ERROR_LENGTH: Final[int] = 2_000

ToolCallable = Callable[[Any], Any]
OnToolUse = Callable[[str, dict[str, Any]], None]
OnToolResult = Callable[[str, str, bool], None]


class AgentTool(Protocol):
    """The uniform tool interface the assistant dispatches against.

    Both local pydantic tools (:class:`ToolSpec`) and remote MCP tools
    (:class:`MCPTool`) satisfy this protocol, so the tool-use loop treats
    them identically: publish ``input_schema()`` to the model, then
    ``run()`` the selected call and place the returned string into the
    ``tool_result`` block.

    ``name`` and ``description`` are declared read-only so that frozen
    dataclasses (whose attributes are immutable) structurally satisfy the
    protocol; a plain ``name: str`` would demand a settable attribute.
    """

    @property
    def name(self) -> str:
        """The tool's unique identifier, as advertised to the model."""
        ...

    @property
    def description(self) -> str:
        """A short natural-language description shown to the model."""
        ...

    def input_schema(self) -> dict[str, Any]:
        """Return the JSON schema advertised to the Anthropic API."""
        ...

    async def run(self, tool_input: dict[str, Any]) -> str:
        """Execute the tool for ``tool_input`` and return its text output."""
        ...


@dataclass(frozen=True)
class ToolSpec:
    """Registration entry that binds a local, pydantic-backed tool to its schema.

    Implements the :class:`AgentTool` protocol: the assistant validates
    the model's raw JSON payload against ``schema`` before handing a typed
    object to ``call``, then serializes whatever ``call`` returns into a
    string for the ``tool_result`` block.
    """

    name: str
    description: str
    schema: type[BaseModel]
    call: ToolCallable

    def input_schema(self) -> dict[str, Any]:
        """Return the JSON schema advertised to the Anthropic API."""
        return self.schema.model_json_schema()

    async def run(self, tool_input: dict[str, Any]) -> str:
        """Validate ``tool_input``, invoke the tool, and serialize its output.

        A :class:`pydantic.ValidationError` raised here propagates to the
        assistant's dispatcher, which surfaces it to the model as an
        ``is_error`` result. Both synchronous and coroutine-returning
        callables are supported transparently.
        """
        validated = self.schema.model_validate(tool_input)
        raw_output = self.call(validated)
        if inspect.isawaitable(raw_output):
            raw_output = await raw_output
        return _serialize_tool_output(raw_output)


@dataclass(frozen=True)
class MCPTool:
    """An MCP-server-provided tool adapted to the :class:`AgentTool` protocol.

    Unlike :class:`ToolSpec`, the input schema is a raw JSON schema dict
    supplied by the server (there is no local pydantic model), and
    execution is delegated to the MCP client, which validates arguments
    server-side and returns already-rendered text.
    """

    name: str
    description: str
    json_schema: dict[str, Any]
    provider: MCPToolProvider

    def input_schema(self) -> dict[str, Any]:
        """Return the server-advertised JSON schema for the tool."""
        return self.json_schema

    async def run(self, tool_input: dict[str, Any]) -> str:
        """Route the call through the MCP client and return its text output."""
        return await self.provider.call_tool(self.name, tool_input)


def _default_tools() -> tuple[ToolSpec, ...]:
    """Return the tool registry the assistant ships with by default."""
    return (
        ToolSpec(
            name="save_note",
            description=(
                "Persist a short markdown note to the local scratch "
                "directory. Use this whenever the user asks you to save, "
                "capture, or write down text they want to keep for later "
                "reference. The tool returns the resolved filename so you "
                "can quote it back to the user."
            ),
            schema=NoteSchema,
            call=save_note,
        ),
        ToolSpec(
            name="search_notes",
            description=(
                "Search across previously saved markdown notes for content "
                "matching a query. Use this whenever the user asks you to "
                "find, recall, look up, or reference something they saved "
                "earlier. Optionally filter by tags. Returns filename, date, "
                "tags, and a short snippet around each match."
            ),
            schema=SearchNotesQuery,
            call=search_notes,
        ),
        ToolSpec(
            name="list_directory",
            description=(
                "List the alphabetized entries of a directory inside the "
                "project workspace. Use this to discover available "
                "workflows, knowledge files, or tools before deciding how "
                "to answer a request. Every path is sandboxed to the "
                "project root; absolute paths and inputs that escape the "
                "root are refused."
            ),
            schema=ListDirectoryQuery,
            call=list_directory,
        ),
        ToolSpec(
            name="read_file",
            description=(
                "Read the UTF-8 contents of a text file inside the project "
                "workspace. Use this to consult a workflow SOP, review a "
                "knowledge document, or inspect any project file before "
                "acting. Every path is sandboxed to the project root; "
                "absolute paths and inputs that escape the root are refused."
            ),
            schema=ReadFileQuery,
            call=read_file,
        ),
        ToolSpec(
            name="web_search",
            description=(
                "Search the public web via DuckDuckGo and return the top "
                "results (title, URL, snippet). Use this when the user "
                "asks a question the knowledge base and local notes cannot "
                "answer, or when they explicitly ask you to look something "
                "up online. Network failures never crash the loop; they "
                "are reported on the result's `error` field."
            ),
            schema=WebSearchQuery,
            call=web_search,
        ),
        ToolSpec(
            name="scrape_url",
            description=(
                "Fetch a URL and return its readable text with HTML noise "
                "stripped. Use this after `web_search` when a snippet is "
                "insufficient and you need the full article body. Network "
                "failures never crash the loop; they are reported on the "
                "result's `error` field."
            ),
            schema=ScrapeUrlQuery,
            call=scrape_url,
        ),
        ToolSpec(
            name="semantic_search",
            description=(
                "Semantic (vector) recall over the Pinecone RAG store. Use "
                "this when the user asks for something previously indexed "
                "from research, scraped articles, or long-form notes — the "
                "match ranking captures meaning, not just keywords. Vector "
                "store failures are reported on the result's `error` field."
            ),
            schema=SemanticSearchQuery,
            call=semantic_search,
        ),
        ToolSpec(
            name="index_document",
            description=(
                "Embed a text body and persist it to the Pinecone RAG "
                "store together with caller-supplied metadata (source URL, "
                "filepath, tags). Use this after a valuable `scrape_url` "
                "or long-form note so future `semantic_search` calls can "
                "recall it. Failures are reported on the `error` field."
            ),
            schema=IndexDocumentQuery,
            call=index_document,
        ),
    )


class PersonalAssistant:
    """Async assistant with a grounded system prompt, multi-turn memory,
    and a bounded tool-use loop."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        tools: Sequence[AgentTool] | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._logger = get_logger(self.__class__.__name__)
        self._client = AsyncAnthropic(api_key=self._settings.anthropic_api_key.get_secret_value())
        registered = tuple(tools) if tools is not None else _default_tools()
        self._tools: dict[str, AgentTool] = {spec.name: spec for spec in registered}
        self._system_prompt: str = self._build_system_prompt()
        self._history: list[MessageParam] = []

    @property
    def history(self) -> list[MessageParam]:
        """Return a defensive copy of the current conversation history."""
        return list(self._history)

    def reset(self) -> None:
        """Discard the accumulated conversation history."""
        self._logger.info(
            "Conversation history cleared (was %d message(s)).",
            len(self._history),
        )
        self._history.clear()

    async def register_mcp_tools(self, provider: MCPToolProvider) -> None:
        """Discover tools from an MCP provider and add them to the registry.

        Construction is synchronous, but MCP discovery is async, so this
        is called after ``__init__`` (typically once, at startup). Each
        advertised tool becomes an :class:`MCPTool` routed through
        ``provider``. A tool whose name collides with an already-registered
        local tool is skipped, so the deterministic local implementation
        always wins. The system prompt is rebuilt only when at least one
        tool is added, so its manifest reflects the new capabilities.
        """
        added = 0
        for info in await provider.list_tools():
            if info.name in self._tools:
                self._logger.warning(
                    "MCP tool %r shadows a registered tool; keeping the local one.",
                    info.name,
                )
                continue
            self._tools[info.name] = MCPTool(
                name=info.name,
                description=info.description,
                json_schema=info.input_schema,
                provider=provider,
            )
            added += 1
        if added:
            self._system_prompt = self._build_system_prompt()
        self._logger.info("Registered %d MCP tool(s).", added)

    async def ask(
        self,
        prompt: str,
        *,
        on_tool_use: OnToolUse | None = None,
        on_tool_result: OnToolResult | None = None,
    ) -> str:
        """Send a user message and return the assistant's final reply.

        Runs the tool-use loop internally. History is only committed to
        `self._history` after the loop terminates successfully, so an
        API exception mid-loop leaves the prior history intact.

        Args:
            prompt: The user's utterance for this turn.
            on_tool_use: Optional callback fired whenever the model
                requests a tool call, receiving `(tool_name, tool_input)`.
            on_tool_result: Optional callback fired after each tool has
                executed (or failed), receiving
                `(tool_name, serialized_output, is_error)`.

        Returns:
            The concatenated text of the final assistant message.
        """
        working: list[MessageParam] = [
            *self._history,
            {"role": "user", "content": prompt},
        ]

        final_text = ""
        for iteration in range(self._settings.max_tool_iterations):
            trimmed = trim_history(working, self._settings.max_history_turns)
            response = await self._client.messages.create(**self._build_create_kwargs(trimmed))

            content_blocks = list(response.content)
            working.append(self._assistant_message(content_blocks))

            stop_reason = getattr(response, "stop_reason", "end_turn")
            if stop_reason != "tool_use":
                final_text = _extract_text(content_blocks)
                self._logger.debug(
                    "Loop finished on iteration %d (stop_reason=%s).",
                    iteration,
                    stop_reason,
                )
                break

            tool_result_blocks = await self._dispatch_tools(
                content_blocks,
                on_tool_use=on_tool_use,
                on_tool_result=on_tool_result,
            )
            working.append({"role": "user", "content": tool_result_blocks})
        else:
            self._logger.warning(
                "Tool-use loop hit the ceiling of %d iterations; returning a graceful message.",
                self._settings.max_tool_iterations,
            )
            final_text = (
                "I hit the tool-use iteration ceiling before producing a "
                "final answer. Please rephrase or narrow the request."
            )
            # Append a synthesized assistant reply so the next turn sees a
            # well-formed history (no dangling user tool_result at the tail).
            working.append({"role": "assistant", "content": final_text})

        self._history = trim_history(working, self._settings.max_history_turns)
        return final_text

    def _build_create_kwargs(self, messages: list[MessageParam]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._settings.anthropic_model,
            "max_tokens": self._settings.anthropic_max_tokens,
            "system": self._system_prompt,
            "messages": messages,
        }
        if self._tools:
            kwargs["tools"] = self._build_tool_definitions()
        return kwargs

    def _build_tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema(),
            }
            for tool in self._tools.values()
        ]

    def _assistant_message(self, blocks: list[Any]) -> MessageParam:
        """Serialize an SDK response into a message the API can echo back.

        Text-only responses are stored as plain strings for readability
        and backward compatibility; anything that carries a `tool_use`
        block is stored as a block list so the pair-preservation
        invariants in `trim_history` can apply.
        """
        if all(getattr(block, "type", None) == "text" for block in blocks):
            return {
                "role": "assistant",
                "content": _extract_text(blocks),
            }
        # Each block is serialized from a dynamic SDK object (or a test
        # double) into a plain dict; the shape is correct but cannot be
        # statically proven, so cast to the API's content-block union.
        serialized = cast(
            "list[ContentBlockParam]",
            [_serialize_block(block) for block in blocks],
        )
        return {
            "role": "assistant",
            "content": serialized,
        }

    async def _dispatch_tools(
        self,
        blocks: Iterable[Any],
        *,
        on_tool_use: OnToolUse | None,
        on_tool_result: OnToolResult | None,
    ) -> list[ToolResultBlockParam]:
        """Execute every `tool_use` block and produce matching `tool_result`s.

        Tool calls are dispatched sequentially in emission order. Both
        synchronous and coroutine-returning tool callables are supported
        transparently: if `spec.call(validated)` returns an awaitable,
        it is awaited before the result is serialized.
        """
        results: list[ToolResultBlockParam] = []
        for block in blocks:
            if getattr(block, "type", None) != "tool_use":
                continue
            results.append(
                await self._invoke_tool(
                    block,
                    on_tool_use=on_tool_use,
                    on_tool_result=on_tool_result,
                )
            )
        return results

    async def _invoke_tool(
        self,
        block: Any,
        *,
        on_tool_use: OnToolUse | None,
        on_tool_result: OnToolResult | None,
    ) -> ToolResultBlockParam:
        """Execute a single tool call, capturing any failure as `is_error`."""
        tool_id = getattr(block, "id", "")
        tool_name = getattr(block, "name", "")
        tool_input = dict(getattr(block, "input", {}) or {})

        if on_tool_use is not None:
            _safe_callback(
                self._logger,
                "on_tool_use",
                lambda: on_tool_use(tool_name, tool_input),
            )

        try:
            tool = self._tools.get(tool_name)
            if tool is None:
                raise ValueError(f"Unknown tool: {tool_name!r}")
            output_str = await tool.run(tool_input)
            self._logger.info("Tool %s returned successfully.", tool_name)
            if on_tool_result is not None:
                _safe_callback(
                    self._logger,
                    "on_tool_result",
                    lambda: on_tool_result(tool_name, output_str, False),
                )
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": output_str,
                "is_error": False,
            }
        except Exception as exc:
            # A tool failure must never crash the loop; it is surfaced to
            # the model as an is_error tool_result so it can self-correct.
            error_str = _format_tool_error(exc)
            self._logger.warning(
                "Tool %s failed (%s); surfacing to model.",
                tool_name,
                type(exc).__name__,
            )
            if on_tool_result is not None:
                _safe_callback(
                    self._logger,
                    "on_tool_result",
                    lambda: on_tool_result(tool_name, error_str, True),
                )
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": error_str,
                "is_error": True,
            }

    def _load_knowledge(self) -> str:
        """Read and concatenate all markdown files from the knowledge directory."""
        knowledge_dir: Path = self._settings.resolved_knowledge_dir
        if not knowledge_dir.exists():
            self._logger.warning("Knowledge directory not found: %s", knowledge_dir)
            return ""

        chunks: list[str] = []
        for path in sorted(knowledge_dir.glob(_KNOWLEDGE_GLOB)):
            try:
                text = path.read_text(encoding="utf-8").strip()
            except OSError as exc:
                self._logger.error("Failed to read %s: %s", path, exc)
                continue
            if text:
                chunks.append(f"# Source: {path.name}\n\n{text}")
                self._logger.debug("Loaded knowledge file: %s", path.name)

        self._logger.info("Loaded %d knowledge file(s).", len(chunks))
        return "\n\n---\n\n".join(chunks)

    def _build_system_prompt(self) -> str:
        """Compose the system prompt injected on every conversation turn.

        The prompt has up to four sections concatenated with blank
        lines: the persona preamble, a dynamically generated tool
        manifest, the tool-use operating protocol, and the knowledge
        base. The tool manifest and protocol are only emitted when at
        least one tool is registered, so an assistant instantiated with
        `tools=()` gets the persona and knowledge alone.
        """
        sections: list[str] = [
            "You are the PlanSmart founder's personal AI assistant. "
            "Always ground your answers in the knowledge base below. "
            "Reply in the language the user writes in (Hungarian or English). "
            "Be concise, direct, and enterprise-grade in tone."
        ]
        if self._tools:
            sections.append(self._build_tool_manifest())
            sections.append(self._build_operating_protocol())

        preamble = "\n\n".join(sections)

        knowledge = self._load_knowledge()
        if not knowledge:
            return preamble
        return f"{preamble}\n\n=== KNOWLEDGE BASE ===\n\n{knowledge}"

    def _build_tool_manifest(self) -> str:
        """Render a compact listing of available tools for the LLM."""
        lines: list[str] = ["Available tools (call only when the user's request warrants it):"]
        for spec in self._tools.values():
            first_sentence = spec.description.split(". ", 1)[0].rstrip(".")
            lines.append(f"  - {spec.name}: {first_sentence}.")
        return "\n".join(lines)

    def _build_operating_protocol(self) -> str:
        """Return the tool-use protocol the assistant must follow."""
        return (
            "Operating protocol:\n"
            "  1. For any multi-step task (briefing, digest, planning), "
            "first call `list_directory` on `workflows/` to see which SOPs "
            "exist, then `read_file` the relevant one and follow its steps "
            "verbatim.\n"
            "  2. When the user asks you to save, capture, or write down "
            "text, call `save_note` rather than describing the action.\n"
            "  3. Never fabricate content that a tool call would reveal — "
            "call the tool. If a tool errors, read the error message and "
            "self-correct on the next iteration.\n"
            "  4. Every tool call must have a purpose derived from the "
            "user's current request. If unsure, ask a clarifying question "
            "before calling any tool."
        )


def _extract_text(blocks: Iterable[Any]) -> str:
    """Concatenate the text of every `type == "text"` block."""
    return "".join(
        getattr(block, "text", "") for block in blocks if getattr(block, "type", None) == "text"
    )


def _serialize_block(block: Any) -> dict[str, Any]:
    """Convert an SDK content block (or a test double) into a plain dict."""
    if hasattr(block, "model_dump"):
        dumped = block.model_dump(mode="python")
        if isinstance(dumped, dict):
            return dumped
    return {key: value for key, value in vars(block).items() if not key.startswith("_")}


def _serialize_tool_output(result: Any) -> str:
    """Render a tool's return value into a string the model can consume."""
    if isinstance(result, BaseModel):
        return result.model_dump_json()
    return str(result)


def _format_tool_error(exc: Exception) -> str:
    """Render an exception into a bounded, model-friendly string."""
    text = f"{type(exc).__name__}: {exc}"
    if len(text) <= _MAX_ERROR_LENGTH:
        return text
    return text[: _MAX_ERROR_LENGTH - 1] + "…"


def _safe_callback(logger: Any, name: str, thunk: Callable[[], None]) -> None:
    """Run a UI callback, swallowing any exception so the loop keeps going."""
    try:
        thunk()
    except Exception:
        logger.exception("Event callback %s raised; continuing.", name)
