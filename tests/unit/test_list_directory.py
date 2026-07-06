"""Unit tests for `tools.list_directory`.

Covers pydantic schema enforcement, happy-path listings inside a
sandboxed `tmp_path`, error-classification edges (missing target, wrong
type), truncation behavior, and — most importantly — that every path-
traversal attack is blocked by the shared sandbox before any file
system access.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tools.list_directory import (
    DEFAULT_MAX_ENTRIES,
    DirEntry,
    ListDirectoryQuery,
    ListDirectoryResult,
    list_directory,
)
from utils.sandbox import SecurityError


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    """Populate `tmp_path` with a small, heterogeneous tree."""
    (tmp_path / "alpha.txt").write_text("a", encoding="utf-8")
    (tmp_path / "beta.md").write_text("b", encoding="utf-8")
    (tmp_path / "gamma").mkdir()
    (tmp_path / "gamma" / "inside.txt").write_text("i", encoding="utf-8")
    return tmp_path


class TestQueryValidation:
    """`ListDirectoryQuery` mirrors the strictness of the other tool schemas."""

    def test_valid_query_round_trips(self) -> None:
        query = ListDirectoryQuery(relative_path=".")
        assert query.relative_path == "."

    def test_empty_string_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ListDirectoryQuery(relative_path="")

    def test_overly_long_path_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ListDirectoryQuery(relative_path="a" * 501)

    def test_extra_fields_are_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            ListDirectoryQuery(relative_path=".", nonsense="drop")  # type: ignore[call-arg]

    def test_query_is_immutable(self) -> None:
        query = ListDirectoryQuery(relative_path=".")
        with pytest.raises(ValidationError):
            query.relative_path = "changed"  # type: ignore[misc]


class TestHappyPath:
    """Valid listings must return sorted, correctly typed entries."""

    def test_root_listing_returns_all_entries(self, sandbox: Path) -> None:
        result = list_directory(
            ListDirectoryQuery(relative_path="."),
            project_root=sandbox,
        )
        assert isinstance(result, ListDirectoryResult)
        assert result.relative_path == "."
        assert result.truncated is False
        assert result.entries == [
            DirEntry(name="alpha.txt", kind="file"),
            DirEntry(name="beta.md", kind="file"),
            DirEntry(name="gamma", kind="directory"),
        ]

    def test_subdirectory_listing_returns_child_entries(self, sandbox: Path) -> None:
        result = list_directory(
            ListDirectoryQuery(relative_path="gamma"),
            project_root=sandbox,
        )
        assert result.relative_path == "gamma"
        assert result.entries == [DirEntry(name="inside.txt", kind="file")]

    def test_dotdot_navigation_that_stays_inside_is_supported(self, sandbox: Path) -> None:
        result = list_directory(
            ListDirectoryQuery(relative_path="gamma/.."),
            project_root=sandbox,
        )
        assert result.relative_path == "."

    def test_empty_directory_returns_empty_entry_list(self, tmp_path: Path) -> None:
        result = list_directory(
            ListDirectoryQuery(relative_path="."),
            project_root=tmp_path,
        )
        assert result.entries == []
        assert result.truncated is False


class TestErrorCases:
    """Non-existent targets and wrong types raise typed exceptions."""

    def test_missing_directory_raises_file_not_found(self, sandbox: Path) -> None:
        with pytest.raises(FileNotFoundError):
            list_directory(
                ListDirectoryQuery(relative_path="does-not-exist"),
                project_root=sandbox,
            )

    def test_listing_a_file_raises_not_a_directory(self, sandbox: Path) -> None:
        with pytest.raises(NotADirectoryError):
            list_directory(
                ListDirectoryQuery(relative_path="alpha.txt"),
                project_root=sandbox,
            )


class TestPathTraversalBlocked:
    """Every attempt to navigate outside the sandbox must fail."""

    def test_dotdot_escape_is_blocked(self, sandbox: Path) -> None:
        with pytest.raises(SecurityError):
            list_directory(
                ListDirectoryQuery(relative_path="../../../etc"),
                project_root=sandbox,
            )

    def test_absolute_path_is_blocked(self, sandbox: Path) -> None:
        with pytest.raises(SecurityError):
            list_directory(
                ListDirectoryQuery(relative_path=str(Path.home())),
                project_root=sandbox,
            )

    def test_null_byte_is_blocked(self, sandbox: Path) -> None:
        with pytest.raises(SecurityError):
            list_directory(
                ListDirectoryQuery(relative_path="valid\x00suffix"),
                project_root=sandbox,
            )


class TestTruncation:
    """Oversized listings must set `truncated=True` and cap entry count."""

    def test_truncation_is_reported_when_entries_exceed_limit(self, tmp_path: Path) -> None:
        for index in range(6):
            (tmp_path / f"file-{index}.txt").write_text("x", encoding="utf-8")
        result = list_directory(
            ListDirectoryQuery(relative_path="."),
            project_root=tmp_path,
            max_entries=3,
        )
        assert result.truncated is True
        assert len(result.entries) == 3

    def test_default_max_entries_constant_is_reasonable(self) -> None:
        assert DEFAULT_MAX_ENTRIES >= 100
