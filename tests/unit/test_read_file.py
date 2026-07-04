"""Unit tests for `tools.read_file`.

Covers pydantic schema enforcement, happy-path reads inside a sandboxed
`tmp_path`, wrong-type / missing-target error classification, byte-limit
truncation, non-UTF-8 fallback decoding, and — most importantly — that
every path-traversal attack is blocked by the shared sandbox before any
file system access.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tools.read_file import (
    DEFAULT_MAX_BYTES,
    ReadFileQuery,
    ReadFileResult,
    read_file,
)
from utils.sandbox import SecurityError


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    """Populate `tmp_path` with a small, heterogeneous tree.

    File content is written via `write_bytes` so line endings are
    preserved verbatim on every OS. `Path.write_text` would apply
    platform-specific newline translation on Windows, which would
    silently corrupt the round-trip assertion below.
    """
    (tmp_path / "notes.md").write_bytes(b"# Heading\n\nSome content.\n")
    (tmp_path / "child").mkdir()
    (tmp_path / "child" / "nested.md").write_bytes(b"nested")
    return tmp_path


class TestQueryValidation:
    """`ReadFileQuery` enforces the same shape rules as the other tool schemas."""

    def test_valid_query_round_trips(self) -> None:
        query = ReadFileQuery(filepath="notes.md")
        assert query.filepath == "notes.md"

    def test_empty_string_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ReadFileQuery(filepath="")

    def test_overly_long_path_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ReadFileQuery(filepath="a" * 501)

    def test_extra_fields_are_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            ReadFileQuery(filepath="notes.md", nonsense="drop")  # type: ignore[call-arg]

    def test_query_is_immutable(self) -> None:
        query = ReadFileQuery(filepath="notes.md")
        with pytest.raises(ValidationError):
            query.filepath = "changed"  # type: ignore[misc]


class TestHappyPath:
    """Valid reads must return the exact content and metadata."""

    def test_read_returns_full_utf8_content(self, sandbox: Path) -> None:
        result = read_file(
            ReadFileQuery(filepath="notes.md"),
            project_root=sandbox,
        )
        assert isinstance(result, ReadFileResult)
        assert result.filepath == "notes.md"
        assert result.content == "# Heading\n\nSome content.\n"
        assert result.truncated is False
        assert result.bytes_read == len(result.content.encode("utf-8"))

    def test_nested_file_is_readable(self, sandbox: Path) -> None:
        result = read_file(
            ReadFileQuery(filepath="child/nested.md"),
            project_root=sandbox,
        )
        assert result.content == "nested"
        assert result.filepath.replace("\\", "/") == "child/nested.md"


class TestErrorCases:
    """Missing files and wrong types raise typed exceptions."""

    def test_missing_file_raises_file_not_found(self, sandbox: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_file(
                ReadFileQuery(filepath="missing.md"),
                project_root=sandbox,
            )

    def test_reading_a_directory_raises_is_a_directory(
        self, sandbox: Path
    ) -> None:
        with pytest.raises(IsADirectoryError):
            read_file(
                ReadFileQuery(filepath="child"),
                project_root=sandbox,
            )


class TestPathTraversalBlocked:
    """Every attempt to read outside the sandbox must fail."""

    def test_dotdot_escape_is_blocked(self, sandbox: Path) -> None:
        with pytest.raises(SecurityError):
            read_file(
                ReadFileQuery(filepath="../../../etc/passwd"),
                project_root=sandbox,
            )

    def test_absolute_path_is_blocked(self, sandbox: Path) -> None:
        with pytest.raises(SecurityError):
            read_file(
                ReadFileQuery(filepath=str(Path.home() / "some.txt")),
                project_root=sandbox,
            )

    def test_null_byte_is_blocked(self, sandbox: Path) -> None:
        with pytest.raises(SecurityError):
            read_file(
                ReadFileQuery(filepath="valid\x00suffix.md"),
                project_root=sandbox,
            )


class TestTruncationAndDecoding:
    """Byte cap must be honored and non-UTF-8 bytes must not crash the tool."""

    def test_oversized_read_is_truncated(self, tmp_path: Path) -> None:
        payload = "x" * 200
        (tmp_path / "big.txt").write_text(payload, encoding="utf-8")
        result = read_file(
            ReadFileQuery(filepath="big.txt"),
            project_root=tmp_path,
            max_bytes=50,
        )
        assert result.truncated is True
        assert result.bytes_read == 50
        assert result.content == "x" * 50

    def test_non_utf8_bytes_are_replaced_not_raised(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "garbage.bin").write_bytes(b"good\xff\xfebad")
        result = read_file(
            ReadFileQuery(filepath="garbage.bin"),
            project_root=tmp_path,
        )
        assert result.content.startswith("good")
        assert "�" in result.content
        assert result.truncated is False

    def test_default_max_bytes_constant_is_reasonable(self) -> None:
        # Ensure the default is large enough for markdown workflows but
        # small enough to fit comfortably in a single LLM context turn.
        assert 10_000 <= DEFAULT_MAX_BYTES <= 1_000_000
