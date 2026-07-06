"""Save a timestamped markdown note to the local scratch directory.

WAT layer: **Tool** — deterministic execution only. No LLM calls, no
reasoning. Every field the LLM proposes is shape-checked and sanitized by
`NoteSchema` before the file system is touched, so a hallucinating model
cannot cause a path traversal, an empty file, an oversized filename, or an
unbounded write.

Notes are written to `.tmp/notes/YYYY-MM-DD-<slug>.md`, which matches the
"intermediates are disposable, deliverables live in the cloud" principle
declared in `knowledge/Claude.md`. If a note with the same date and slug
already exists, a numeric suffix (`-1`, `-2`, ...) is appended so that no
prior note is ever silently overwritten.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.config import PROJECT_ROOT
from utils.logger import get_logger

__all__ = [
    "DEFAULT_NOTES_DIR",
    "NoteSchema",
    "SaveNoteResult",
    "save_note",
]

_logger = get_logger(__name__)

DEFAULT_NOTES_DIR: Final[Path] = PROJECT_ROOT / ".tmp" / "notes"

_SLUG_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_MAX_SLUG_LENGTH: Final[int] = 60
_MAX_CONTENT_LENGTH: Final[int] = 20_000
_MAX_TAGS: Final[int] = 10


class NoteSchema(BaseModel):
    """Structured tool input, enforced before the file system is touched.

    Every field is bound by an explicit regex or length limit. The model is
    `frozen` so callers cannot mutate a validated payload behind the tool's
    back, and `extra="forbid"` prevents the LLM from smuggling unexpected
    keys through the tool interface.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        frozen=True,
    )

    slug: str = Field(
        ...,
        min_length=1,
        max_length=_MAX_SLUG_LENGTH,
        description=(
            "URL-safe kebab-case identifier used to build the filename. "
            "Must match ^[a-z0-9]+(-[a-z0-9]+)*$."
        ),
    )
    content: str = Field(
        ...,
        min_length=1,
        max_length=_MAX_CONTENT_LENGTH,
        description="Markdown body of the note; stored verbatim.",
    )
    tags: list[str] = Field(
        default_factory=list,
        max_length=_MAX_TAGS,
        description=(
            "Optional list of short tags recorded as YAML frontmatter. "
            "Each tag must be lowercase kebab-case."
        ),
    )

    @field_validator("slug")
    @classmethod
    def _slug_is_kebab_case(cls, value: str) -> str:
        if not _SLUG_PATTERN.fullmatch(value):
            raise ValueError("slug must be lowercase kebab-case, e.g. 'q3-planning-notes'")
        return value

    @field_validator("tags")
    @classmethod
    def _tags_are_kebab_case(cls, values: list[str]) -> list[str]:
        cleaned = [tag.strip().lower() for tag in values if tag.strip()]
        for tag in cleaned:
            if not _SLUG_PATTERN.fullmatch(tag):
                raise ValueError(f"tag {tag!r} must be lowercase kebab-case")
        return cleaned


class SaveNoteResult(BaseModel):
    """Metadata returned to the Agent layer after a successful write."""

    model_config = ConfigDict(frozen=True)

    path: Path = Field(..., description="Absolute path of the written note.")
    filename: str = Field(..., description="Basename of the written file.")
    bytes_written: int = Field(..., ge=0)


def save_note(
    note: NoteSchema,
    *,
    base_dir: Path | None = None,
    today: date | None = None,
) -> SaveNoteResult:
    """Persist `note` as a markdown file and return metadata about the write.

    The target directory is created on demand. If a file with the same date
    and slug already exists, a numeric suffix (`-1`, `-2`, ...) is appended
    so that no prior note is ever silently overwritten.

    Args:
        note: Pre-validated tool input. Callers must convert an LLM payload
            via `NoteSchema.model_validate(...)` before invocation.
        base_dir: Directory to write into. Defaults to `.tmp/notes/` under
            the project root; overridable in tests.
        today: Date used to prefix the filename. Defaults to `date.today()`;
            overridable in tests for deterministic assertions.

    Returns:
        `SaveNoteResult` with the resolved path, basename, and byte count.

    Raises:
        ValueError: If the resolved target escapes `base_dir` (defense in
            depth against symlinked or otherwise adversarial base paths).
        OSError: If the file system rejects the write (permissions, disk
            full, read-only mount). Propagated so the Agent can react.
    """
    root = (base_dir if base_dir is not None else DEFAULT_NOTES_DIR).resolve()
    root.mkdir(parents=True, exist_ok=True)

    stamp = (today or date.today()).isoformat()
    target = _next_available_path(root, stamp, note.slug)

    resolved = target.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Refusing to write outside {root}: {resolved}")

    encoded = _render(note, stamp).encode("utf-8")
    target.write_bytes(encoded)

    _logger.info("Saved note %s (%d bytes).", target, len(encoded))
    return SaveNoteResult(
        path=resolved,
        filename=resolved.name,
        bytes_written=len(encoded),
    )


def _next_available_path(root: Path, stamp: str, slug: str) -> Path:
    """Return a non-colliding path in `root` for the given date and slug."""
    candidate = root / f"{stamp}-{slug}.md"
    if not candidate.exists():
        return candidate
    counter = 1
    while True:
        candidate = root / f"{stamp}-{slug}-{counter}.md"
        if not candidate.exists():
            return candidate
        counter += 1


def _render(note: NoteSchema, stamp: str) -> str:
    """Format the note as markdown with a YAML frontmatter block."""
    lines: list[str] = ["---", f"date: {stamp}", f"slug: {note.slug}"]
    if note.tags:
        lines.append("tags:")
        lines.extend(f"  - {tag}" for tag in note.tags)
    lines.append("---")
    lines.append("")
    lines.append(note.content.strip())
    lines.append("")
    return "\n".join(lines)
