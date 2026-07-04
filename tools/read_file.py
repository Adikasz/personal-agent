"""Read a text file from inside the project workspace.

WAT layer: **Tool** — deterministic execution only. No LLM calls, no
reasoning. Every path proposed by the LLM is validated by
`utils.sandbox.resolve_within_project` before the file system is
touched, so a hallucinating model cannot read a file above the project
root.

Reads are bounded by `DEFAULT_MAX_BYTES` to protect the LLM's context
window; oversized files are truncated and the returned metadata sets
`truncated=True` so the model can either narrow its request or ask the
user for guidance. Non-UTF-8 bytes are decoded with `errors="replace"`
so a binary file drop does not crash the loop — the model simply sees
replacement characters and can decide how to react.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from utils.logger import get_logger
from utils.sandbox import PROJECT_ROOT, resolve_within_project

__all__ = [
    "DEFAULT_MAX_BYTES",
    "ReadFileQuery",
    "ReadFileResult",
    "read_file",
]

_logger = get_logger(__name__)

DEFAULT_MAX_BYTES: Final[int] = 100_000
_MAX_PATH_LENGTH: Final[int] = 500


class ReadFileQuery(BaseModel):
    """Structured tool input, enforced before any file system access."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        frozen=True,
    )

    filepath: str = Field(
        ...,
        min_length=1,
        max_length=_MAX_PATH_LENGTH,
        description=(
            "Path to the file, relative to the project root. Absolute "
            "paths and inputs that escape the project root are refused "
            "with a SecurityError."
        ),
    )


class ReadFileResult(BaseModel):
    """Metadata returned to the Agent layer after a successful read."""

    model_config = ConfigDict(frozen=True)

    filepath: str
    content: str
    bytes_read: int = Field(..., ge=0)
    truncated: bool


def read_file(
    query: ReadFileQuery,
    *,
    project_root: Path | None = None,
    max_bytes: int | None = None,
) -> ReadFileResult:
    """Return the UTF-8 contents of `query.filepath`.

    Args:
        query: Pre-validated tool input. Callers must convert an LLM
            payload via `ReadFileQuery.model_validate(...)` before
            invocation.
        project_root: Test-only sandbox root override. Production callers
            must never pass this argument.
        max_bytes: Test-only override for the read cap.

    Returns:
        A `ReadFileResult` with the resolved relative path, the decoded
        content, the byte count actually read, and a `truncated` flag.

    Raises:
        SecurityError: If the path escapes the project root.
        FileNotFoundError: If the resolved target does not exist.
        IsADirectoryError: If the resolved target exists but is a directory.
        OSError: For other underlying filesystem failures.
    """
    resolved = resolve_within_project(query.filepath, project_root=project_root)
    if not resolved.exists():
        raise FileNotFoundError(
            f"{query.filepath!r} does not exist inside the project root."
        )
    if resolved.is_dir():
        raise IsADirectoryError(
            f"{query.filepath!r} is a directory; use list_directory instead."
        )

    limit = max_bytes if max_bytes is not None else DEFAULT_MAX_BYTES
    size = resolved.stat().st_size
    read_size = min(size, limit)
    truncated = size > limit

    with resolved.open("rb") as handle:
        raw = handle.read(read_size)

    content = raw.decode("utf-8", errors="replace")

    display_root = (project_root if project_root is not None else PROJECT_ROOT).resolve()
    display_path = str(resolved.relative_to(display_root))

    _logger.info(
        "Read %d/%d byte(s) from %s (truncated=%s).",
        len(raw),
        size,
        display_path,
        truncated,
    )
    return ReadFileResult(
        filepath=display_path,
        content=content,
        bytes_read=len(raw),
        truncated=truncated,
    )
