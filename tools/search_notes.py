"""Search across the local markdown notes scratchpad.

WAT layer: **Tool** — deterministic execution only. No LLM calls, no
reasoning. Reads the same directory that `tools.save_note.save_note`
writes into (`.tmp/notes/` by default) and returns structured matches
so the Agent layer can decide how to present them to the user.

Both the search query and the tag filter are enforced by pydantic
before any file system access, so a hallucinating LLM cannot smuggle
oversized inputs, path-traversal characters, or malformed tags into the
tool. A YAML frontmatter parser tailored to the exact shape produced by
`save_note` extracts the date, slug, and tags without pulling in a full
YAML dependency.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tools.save_note import DEFAULT_NOTES_DIR
from utils.logger import get_logger

__all__ = [
    "NoteMatch",
    "SearchNotesQuery",
    "SearchNotesResult",
    "search_notes",
]

_logger = get_logger(__name__)

_TAG_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_MAX_QUERY_LENGTH: Final[int] = 200
_MAX_TAGS: Final[int] = 10
_SNIPPET_CONTEXT_CHARS: Final[int] = 80
_FRONTMATTER_DELIMITER: Final[str] = "---\n"


class SearchNotesQuery(BaseModel):
    """Structured tool input for `search_notes`.

    Enforced by pydantic before the file system is touched. `extra="forbid"`
    prevents the LLM from smuggling unexpected keys, and the frozen model
    guarantees a validated payload cannot be mutated behind the tool's back.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        frozen=True,
    )

    query: str = Field(
        ...,
        min_length=1,
        max_length=_MAX_QUERY_LENGTH,
        description=(
            "Substring to search for inside note bodies, matched "
            "case-insensitively. Whitespace-only queries are rejected."
        ),
    )
    tags: list[str] = Field(
        default_factory=list,
        max_length=_MAX_TAGS,
        description=(
            "Optional tag filter. When present, only notes whose "
            "frontmatter contains every listed tag are considered."
        ),
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Upper bound on the number of matches to return.",
    )

    @field_validator("query")
    @classmethod
    def _query_is_non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must contain at least one non-whitespace character")
        return value

    @field_validator("tags")
    @classmethod
    def _tags_are_kebab_case(cls, values: list[str]) -> list[str]:
        cleaned = [tag.strip().lower() for tag in values if tag.strip()]
        for tag in cleaned:
            if not _TAG_PATTERN.fullmatch(tag):
                raise ValueError(f"tag {tag!r} must be lowercase kebab-case")
        return cleaned


class NoteMatch(BaseModel):
    """A single note that satisfied the search predicate."""

    model_config = ConfigDict(frozen=True)

    filename: str
    date: str
    slug: str
    tags: list[str]
    snippet: str


class SearchNotesResult(BaseModel):
    """Structured return value handed back to the Agent layer."""

    model_config = ConfigDict(frozen=True)

    query: str
    scanned: int = Field(..., ge=0)
    matches: list[NoteMatch]


def search_notes(
    query: SearchNotesQuery,
    *,
    base_dir: Path | None = None,
) -> SearchNotesResult:
    """Return notes whose body contains `query.query` (case-insensitive).

    The scratchpad directory is scanned in filename order (which, by
    virtue of the `YYYY-MM-DD` prefix produced by `save_note`, is
    chronological). Scanning stops as soon as `query.limit` matches
    have been collected so a large history does not blow up latency.

    Args:
        query: Pre-validated tool input. The caller must convert an LLM
            payload via `SearchNotesQuery.model_validate(...)` before
            invocation.
        base_dir: Directory to search in. Defaults to `.tmp/notes/`
            under the project root; overridable in tests.

    Returns:
        A `SearchNotesResult` with the match list, the count of files
        actually scanned, and the echoed query for context.
    """
    root = (base_dir if base_dir is not None else DEFAULT_NOTES_DIR).resolve()
    if not root.exists():
        _logger.info("Notes directory %s does not exist yet; no matches.", root)
        return SearchNotesResult(query=query.query, scanned=0, matches=[])

    needle = query.query.lower()
    required_tags = frozenset(query.tags)
    matches: list[NoteMatch] = []
    scanned = 0

    for path in sorted(root.glob("*.md")):
        scanned += 1
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            _logger.warning("Skipping unreadable note %s: %s", path, exc)
            continue

        metadata, body = _split_frontmatter(text)
        note_tags = _coerce_tag_list(metadata.get("tags"))

        if required_tags and not required_tags.issubset(note_tags):
            continue

        haystack = body.lower()
        hit = haystack.find(needle)
        if hit == -1:
            continue

        matches.append(
            NoteMatch(
                filename=path.name,
                date=str(metadata.get("date", "")),
                slug=str(metadata.get("slug", "")),
                tags=note_tags,
                snippet=_snippet(body, hit),
            )
        )
        if len(matches) >= query.limit:
            break

    _logger.info(
        "Search for %r matched %d of %d note(s).",
        query.query,
        len(matches),
        scanned,
    )
    return SearchNotesResult(
        query=query.query,
        scanned=scanned,
        matches=matches,
    )


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Separate a YAML frontmatter block from the note body.

    Recognizes only the exact shape produced by `save_note`. Anything
    else is returned as body-only with empty metadata rather than
    raising, so third-party notes dropped into the directory degrade
    gracefully instead of poisoning the search.
    """
    if not text.startswith(_FRONTMATTER_DELIMITER):
        return {}, text
    end = text.find(f"\n{_FRONTMATTER_DELIMITER}", len(_FRONTMATTER_DELIMITER))
    if end == -1:
        return {}, text
    yaml_block = text[len(_FRONTMATTER_DELIMITER) : end]
    body_start = end + len(_FRONTMATTER_DELIMITER) + 1  # skip the closing "\n---\n"
    return _parse_frontmatter(yaml_block), text[body_start:]


def _parse_frontmatter(block: str) -> dict[str, Any]:
    """Parse the narrow YAML shape emitted by `save_note`.

    Supports two forms:

    - `key: value` scalar assignment.
    - `key:` followed by one or more `  - item` list entries.

    Every other construct is ignored, which keeps this parser well
    below the surface area of a real YAML library.
    """
    result: dict[str, Any] = {}
    active_list_key: str | None = None
    for raw_line in block.splitlines():
        if raw_line.startswith("  - "):
            if active_list_key is not None:
                result.setdefault(active_list_key, []).append(raw_line[4:].strip())
            continue

        active_list_key = None
        stripped = raw_line.strip()
        if not stripped or ":" not in stripped:
            continue

        key, _, value = stripped.partition(":")
        key = key.strip()
        value = value.strip()
        if not value:
            active_list_key = key
            result.setdefault(key, [])
        else:
            result[key] = value
    return result


def _coerce_tag_list(value: Any) -> list[str]:
    """Normalize a frontmatter tag value into a clean list of strings."""
    if isinstance(value, list):
        return [str(item).strip().lower() for item in value if str(item).strip()]
    return []


def _snippet(body: str, hit: int) -> str:
    """Return a short human-readable excerpt centered on the match."""
    start = max(0, hit - _SNIPPET_CONTEXT_CHARS)
    end = min(len(body), hit + _SNIPPET_CONTEXT_CHARS)
    excerpt = body[start:end].strip()
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(body) else ""
    return f"{prefix}{excerpt}{suffix}"
