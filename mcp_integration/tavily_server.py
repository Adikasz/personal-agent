"""Tavily search exposed as a stdio MCP server.

WAT layer: **Tool** — deterministic execution with no reasoning of its
own. The module wraps a single call to the ``tavily-python`` SDK behind
the MCP ``@mcp.tool()`` decorator and renders the response into a clean,
LLM-ready summary. It runs as its own process: the ``TavilyMCPClient``
spawns it over stdio, so the module's ``main()`` entry point simply
starts the server on the stdio transport.

Network and configuration failures are **never** raised out of the tool.
They are formatted into the returned string (prefixed with a clear
marker) so the model sees the failure as ordinary tool output and can
pivot, exactly as the local ``web_search`` tool does. Importing this
module performs no I/O and constructs no Tavily client, so the test
suite can exercise the search logic offline by injecting a fake client.
"""

from __future__ import annotations

import os
from typing import Any, Final, Literal

from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient

_API_KEY_ENV: Final[str] = "TAVILY_API_KEY"
DEFAULT_MAX_RESULTS: Final[int] = 5
_ABSOLUTE_MAX_RESULTS: Final[int] = 20
_MAX_CONTENT_CHARS: Final[int] = 600
_SEARCH_DEPTH: Final[str] = "basic"
DEFAULT_TOPIC: Final[str] = "general"
DEFAULT_NEWS_DAYS: Final[int] = 7
_MAX_NEWS_DAYS: Final[int] = 30
_VALID_TOPICS: Final[tuple[str, ...]] = ("general", "news")

mcp = FastMCP(
    "plansmart-tavily",
    instructions=(
        "Provides a single web-search tool backed by the Tavily API. Call "
        "`tavily_search` for fresh, high-signal answers to questions the "
        "local knowledge base cannot cover."
    ),
)


def _build_client() -> TavilyClient:
    """Construct a Tavily client from the subprocess environment.

    The key is passed in by ``TavilyMCPClient`` when it spawns this
    server. A missing key raises, which the caller converts into a
    tool-output error string rather than a crash.
    """
    api_key = os.environ.get(_API_KEY_ENV, "").strip()
    if not api_key:
        raise RuntimeError(f"{_API_KEY_ENV} is not set in the MCP server environment.")
    return TavilyClient(api_key=api_key)


def _clip(text: str, limit: int = _MAX_CONTENT_CHARS) -> str:
    """Collapse whitespace and bound a snippet to ``limit`` characters."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"


def _format_results(payload: dict[str, Any], query: str) -> str:
    """Render a Tavily response dict into a compact, model-friendly digest."""
    lines: list[str] = [f"Search results for: {query}"]

    answer = str(payload.get("answer") or "").strip()
    if answer:
        lines.append(f"\nSummary: {_clip(answer)}")

    results = payload.get("results")
    rows: list[dict[str, Any]] = results if isinstance(results, list) else []
    if not rows:
        lines.append("\nNo results were returned for this query.")
        return "\n".join(lines)

    lines.append("")
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        title = str(row.get("title") or "Untitled").strip()
        url = str(row.get("url") or "").strip()
        content = _clip(str(row.get("content") or "").strip())
        lines.append(f"{index}. {title}")
        if url:
            lines.append(f"   URL: {url}")
        if content:
            lines.append(f"   {content}")
    return "\n".join(lines)


def _search(
    query: str,
    max_results: int,
    topic: str = DEFAULT_TOPIC,
    days: int = DEFAULT_NEWS_DAYS,
    *,
    client: TavilyClient | None = None,
) -> str:
    """Execute a Tavily search and return an LLM-ready summary.

    Args:
        query: The user's search query.
        max_results: Requested result count; clamped to ``[1, 20]``.
        topic: Either ``"general"`` (default) or ``"news"``. In news mode
            the ``topic`` and ``days`` parameters are forwarded to Tavily
            so only strictly recent articles are returned; in general
            mode neither is sent, preserving the original behavior.
        days: Recency window in days for news mode; clamped to
            ``[1, 30]``. Ignored when ``topic`` is ``"general"``.
        client: Test-only injection point for a fake Tavily client. When
            omitted, a real client is built from the environment.

    Returns:
        A formatted summary string, or an error string prefixed with
        ``"Tavily search failed:"`` when the search could not complete.
    """
    trimmed = query.strip()
    if not trimmed:
        return "Tavily search failed: query must contain non-whitespace text."
    if topic not in _VALID_TOPICS:
        return f"Tavily search failed: unsupported topic {topic!r}; use 'general' or 'news'."

    capped = max(1, min(max_results, _ABSOLUTE_MAX_RESULTS))
    search_kwargs: dict[str, Any] = {
        "max_results": capped,
        "search_depth": _SEARCH_DEPTH,
        "include_answer": True,
    }
    if topic == "news":
        search_kwargs["topic"] = "news"
        search_kwargs["days"] = max(1, min(days, _MAX_NEWS_DAYS))

    try:
        active = client if client is not None else _build_client()
        payload = active.search(trimmed, **search_kwargs)
    except Exception as exc:
        # A search or configuration failure is surfaced to the model as
        # ordinary tool output, never raised, mirroring the local tools.
        return f"Tavily search failed: {type(exc).__name__}: {exc}"

    if not isinstance(payload, dict):
        return "Tavily search failed: unexpected response shape from Tavily."
    return _format_results(payload, trimmed)


@mcp.tool()
def tavily_search(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    topic: Literal["general", "news"] = "general",
    days: int = DEFAULT_NEWS_DAYS,
) -> str:
    """Search the public web via Tavily and return ranked, summarized results.

    Use this when the user asks about current events, external facts, or
    anything the local knowledge base and notes cannot answer. Returns a
    short synthesized summary followed by the top sources (title, URL,
    and a snippet). For time-sensitive questions (breaking news, "what
    happened this week", latest releases) pass ``topic="news"`` and set
    ``days`` to the recency window you need — this forces strictly
    recent articles instead of evergreen pages. Search or configuration
    failures come back as a string beginning with "Tavily search
    failed:" — never as an exception — so read that message and adjust
    rather than retrying blindly.

    Args:
        query: The natural-language search query.
        max_results: Maximum number of source results to return
            (1 to 20, default 5).
        topic: "general" for evergreen web search (default) or "news"
            for strictly recent news coverage.
        days: How many days back news results may reach (1 to 30,
            default 7). Only applies when topic is "news".
    """
    return _search(query, max_results, topic, days)


def main() -> None:
    """Run the Tavily MCP server on the stdio transport."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
