"""List the entries of a directory inside the project workspace.

WAT layer: **Tool** — deterministic execution only. No LLM calls, no
reasoning. Every path proposed by the LLM is validated by
`utils.sandbox.resolve_within_project` before the file system is
touched, so a hallucinating model cannot navigate above the project
root.

The tool returns a small, well-typed structure (`ListDirectoryResult`)
containing each entry's basename and whether it is a file or a
directory, ordered alphabetically. Results are capped at
`DEFAULT_MAX_ENTRIES` so a pathological directory (thousands of build
artefacts, for example) cannot exhaust the LLM's context window; the
`truncated` field signals the LLM to narrow its query when this happens.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from utils.logger import get_logger
from utils.sandbox import PROJECT_ROOT, resolve_within_project

__all__ = [
    "DEFAULT_MAX_ENTRIES",
    "DirEntry",
    "ListDirectoryQuery",
    "ListDirectoryResult",
    "list_directory",
]

_logger = get_logger(__name__)

DEFAULT_MAX_ENTRIES: Final[int] = 500
_MAX_PATH_LENGTH: Final[int] = 500


class ListDirectoryQuery(BaseModel):
    """Structured tool input, enforced before any file system access."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        frozen=True,
    )

    relative_path: str = Field(
        ...,
        min_length=1,
        max_length=_MAX_PATH_LENGTH,
        description=(
            "Path to list, relative to the project root. Use '.' for the "
            "root itself. Absolute paths and inputs that escape the "
            "project root are refused with a SecurityError."
        ),
    )


class DirEntry(BaseModel):
    """A single entry inside the listed directory."""

    model_config = ConfigDict(frozen=True)

    name: str
    kind: Literal["file", "directory"]


class ListDirectoryResult(BaseModel):
    """Metadata returned to the Agent layer after a successful listing."""

    model_config = ConfigDict(frozen=True)

    relative_path: str
    entries: list[DirEntry]
    truncated: bool


def list_directory(
    query: ListDirectoryQuery,
    *,
    project_root: Path | None = None,
    max_entries: int | None = None,
) -> ListDirectoryResult:
    """Return the alphabetized contents of `query.relative_path`.

    Args:
        query: Pre-validated tool input. Callers must convert an LLM
            payload via `ListDirectoryQuery.model_validate(...)` before
            invocation.
        project_root: Test-only sandbox root override. Production callers
            must never pass this argument.
        max_entries: Test-only override for the entry cap.

    Returns:
        A `ListDirectoryResult` with the resolved relative path, the
        entry list, and a `truncated` flag.

    Raises:
        SecurityError: If the path escapes the project root.
        FileNotFoundError: If the resolved target does not exist.
        NotADirectoryError: If the resolved target exists but is not a
            directory.
    """
    resolved = resolve_within_project(query.relative_path, project_root=project_root)
    if not resolved.exists():
        raise FileNotFoundError(f"{query.relative_path!r} does not exist inside the project root.")
    if not resolved.is_dir():
        raise NotADirectoryError(f"{query.relative_path!r} is not a directory.")

    limit = max_entries if max_entries is not None else DEFAULT_MAX_ENTRIES
    entries: list[DirEntry] = []
    truncated = False
    for child in sorted(resolved.iterdir(), key=lambda item: item.name):
        if len(entries) >= limit:
            truncated = True
            break
        entries.append(
            DirEntry(
                name=child.name,
                kind="directory" if child.is_dir() else "file",
            )
        )

    display_root = (project_root if project_root is not None else PROJECT_ROOT).resolve()
    relative = resolved.relative_to(display_root)
    display_path = str(relative) if str(relative) != "." else "."

    _logger.info(
        "Listed %d entr(y|ies) in %s (truncated=%s).",
        len(entries),
        display_path,
        truncated,
    )
    return ListDirectoryResult(
        relative_path=display_path,
        entries=entries,
        truncated=truncated,
    )
