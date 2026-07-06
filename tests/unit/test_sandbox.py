"""Unit tests for `utils.sandbox`.

These tests are the enforcement mechanism for the project's filesystem
security boundary. Every path-traversal or absolute-path attack must
fail loudly with `SecurityError`, and every legitimate relative path
that resolves back inside the sandbox root must succeed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from utils.sandbox import PROJECT_ROOT, SecurityError, resolve_within_project


class TestValidInputs:
    """Well-formed relative paths inside the sandbox must resolve normally."""

    def test_dot_resolves_to_the_sandbox_root(self, tmp_path: Path) -> None:
        resolved = resolve_within_project(".", project_root=tmp_path)
        assert resolved == tmp_path.resolve()

    def test_relative_child_path_is_allowed(self, tmp_path: Path) -> None:
        (tmp_path / "child").mkdir()
        resolved = resolve_within_project("child", project_root=tmp_path)
        assert resolved == (tmp_path / "child").resolve()

    def test_nested_relative_path_is_allowed(self, tmp_path: Path) -> None:
        (tmp_path / "a" / "b").mkdir(parents=True)
        resolved = resolve_within_project("a/b", project_root=tmp_path)
        assert resolved == (tmp_path / "a" / "b").resolve()

    def test_dotdot_that_stays_inside_is_allowed(self, tmp_path: Path) -> None:
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        resolved = resolve_within_project("a/../b", project_root=tmp_path)
        assert resolved == (tmp_path / "b").resolve()

    def test_accepts_path_object_input(self, tmp_path: Path) -> None:
        (tmp_path / "child").mkdir()
        resolved = resolve_within_project(Path("child"), project_root=tmp_path)
        assert resolved == (tmp_path / "child").resolve()

    def test_default_project_root_is_the_workspace_root(self) -> None:
        # No `project_root=` override — must sandbox against the real
        # project root. `.` must resolve to that root.
        resolved = resolve_within_project(".")
        assert resolved == PROJECT_ROOT.resolve()


class TestPathTraversalAttacks:
    """Every attempt to escape the sandbox must raise SecurityError."""

    def test_dotdot_that_escapes_root_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(SecurityError):
            resolve_within_project("../../../etc/passwd", project_root=tmp_path)

    def test_repeated_dotdot_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(SecurityError):
            resolve_within_project("../..", project_root=tmp_path)

    def test_dotdot_after_a_legitimate_segment_still_escapes(self, tmp_path: Path) -> None:
        with pytest.raises(SecurityError):
            resolve_within_project("a/../../outside", project_root=tmp_path)


class TestAbsolutePathsAreRefused:
    """Regardless of the underlying OS, an absolute path always fails."""

    def test_home_directory_path_is_refused(self) -> None:
        # `Path.home()` is absolute on every supported OS, which makes
        # this assertion cross-platform.
        with pytest.raises(SecurityError):
            resolve_within_project(str(Path.home()))

    def test_project_root_expressed_as_absolute_is_still_refused(self, tmp_path: Path) -> None:
        # Even an absolute path that *would* resolve inside the sandbox
        # is refused: the LLM's contract is "relative only."
        with pytest.raises(SecurityError):
            resolve_within_project(str(tmp_path), project_root=tmp_path)


class TestMalformedInputs:
    """Empty and null-byte inputs must raise SecurityError before I/O."""

    def test_empty_string_is_refused(self) -> None:
        with pytest.raises(SecurityError):
            resolve_within_project("")

    def test_null_byte_is_refused(self) -> None:
        with pytest.raises(SecurityError):
            resolve_within_project("foo\x00bar")


class TestSymlinkEscape:
    """A symlink that points outside the sandbox must be caught."""

    def test_symlink_leading_out_of_the_root_is_refused(self, tmp_path: Path) -> None:
        # Build two sibling roots: the sandbox, and an "outside" area.
        sandbox = tmp_path / "sandbox"
        outside = tmp_path / "outside"
        sandbox.mkdir()
        outside.mkdir()
        target = outside / "secret.txt"
        target.write_text("secret", encoding="utf-8")

        link = sandbox / "escape"
        try:
            link.symlink_to(target)
        except (OSError, NotImplementedError):
            pytest.skip("Symbolic-link creation is unavailable in this environment.")

        with pytest.raises(SecurityError):
            resolve_within_project("escape", project_root=sandbox)
