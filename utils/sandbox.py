"""Filesystem sandbox for workspace-aware tools.

WAT layer: **Tool** — pure computation, no LLM calls, no I/O. Every
tool that accepts a path from the LLM must route that path through
`resolve_within_project` before touching the disk. The function is the
single chokepoint that guarantees a hallucinating or adversarial input
cannot escape the project root via `..` segments, absolute paths,
null-byte injection, or symbolic-link redirection.

The sandbox is intentionally *deny by default*:

- Empty strings are refused.
- Null bytes are refused.
- Absolute paths (whether Windows drive-letter or POSIX) are refused
  outright, so the LLM's contract is unambiguous: "supply a path
  relative to the project root, or use '.' for the root itself."
- Relative paths whose fully-resolved target — after evaluating every
  `..` segment and following symlinks — falls outside the project root
  raise a `SecurityError`.

Paths whose input contains `..` but resolve *back inside* the root are
allowed. The policy enforces the property "resolved target lives inside
the root," not the syntactic heuristic "input does not contain '..'",
which would either false-positive on legitimate navigation or
false-negative on symlink attacks.
"""

from __future__ import annotations

from pathlib import Path

from core.config import PROJECT_ROOT

__all__ = ["PROJECT_ROOT", "SecurityError", "resolve_within_project"]


class SecurityError(Exception):
    """Raised when a filesystem tool is asked to operate outside the project root."""


def resolve_within_project(
    candidate: str | Path,
    *,
    project_root: Path | None = None,
) -> Path:
    """Resolve `candidate` and prove it lives inside the project root.

    Args:
        candidate: Path proposed by the LLM. May be a plain string or an
            already-constructed `Path`. It is treated as relative to the
            project root; absolute inputs are refused.
        project_root: Override for the sandboxed root, used exclusively
            in tests to point at a `tmp_path` fixture. Production callers
            must never pass this argument.

    Returns:
        The fully resolved absolute `Path` corresponding to `candidate`,
        guaranteed to live under the (resolved) project root.

    Raises:
        SecurityError: If `candidate` is empty, contains a null byte, is
            absolute, or resolves to a location outside the project root.
    """
    text = "" if candidate is None else str(candidate)
    if not text:
        raise SecurityError("Empty path is not allowed.")
    if "\x00" in text:
        raise SecurityError("Null byte in path is not allowed.")

    supplied = Path(text)
    if supplied.is_absolute():
        raise SecurityError(
            f"Absolute paths are refused; supply a path relative to the project root. Got: {text!r}"
        )

    root = (project_root if project_root is not None else PROJECT_ROOT).resolve()
    resolved = (root / supplied).resolve()
    if not resolved.is_relative_to(root):
        raise SecurityError(f"Path {text!r} escapes the project root {root!s}.")
    return resolved
