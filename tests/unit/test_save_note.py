"""Unit tests for `tools.save_note`.

These tests exercise the pydantic validation layer and the deterministic
file-system side effects in isolation, using `tmp_path` so no test ever
writes into the real `.tmp/notes/` directory.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from tools.save_note import NoteSchema, SaveNoteResult, save_note


@pytest.fixture
def fixed_today() -> date:
    """A frozen date so filename assertions are deterministic."""
    return date(2026, 7, 4)


class TestNoteSchemaValidation:
    """`NoteSchema` is the LLM-facing contract; every field is strictly bounded."""

    def test_valid_payload_round_trips(self) -> None:
        note = NoteSchema(
            slug="q3-planning",
            content="Body of the note.",
            tags=["planning", "q3"],
        )
        assert note.slug == "q3-planning"
        assert note.tags == ["planning", "q3"]

    def test_default_tags_are_empty(self) -> None:
        note = NoteSchema(slug="s", content="c")
        assert note.tags == []

    def test_uppercase_slug_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            NoteSchema(slug="Not-Kebab", content="c")

    def test_slug_with_spaces_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            NoteSchema(slug="two words", content="c")

    def test_slug_with_path_traversal_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            NoteSchema(slug="../etc/passwd", content="c")

    def test_slug_with_leading_or_trailing_hyphen_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            NoteSchema(slug="-leading", content="c")
        with pytest.raises(ValidationError):
            NoteSchema(slug="trailing-", content="c")

    def test_overly_long_slug_is_rejected(self) -> None:
        long_slug = "a" * 61
        with pytest.raises(ValidationError):
            NoteSchema(slug=long_slug, content="c")

    def test_empty_content_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            NoteSchema(slug="s", content="")

    def test_whitespace_only_content_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            NoteSchema(slug="s", content="   ")

    def test_overly_long_content_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            NoteSchema(slug="s", content="x" * 20_001)

    def test_extra_fields_are_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            NoteSchema(slug="s", content="c", nonsense="drop me")  # type: ignore[call-arg]

    def test_tags_are_lowercased_and_stripped(self) -> None:
        note = NoteSchema(slug="s", content="c", tags=["  Planning  ", "Q3"])
        assert note.tags == ["planning", "q3"]

    def test_tag_with_invalid_characters_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            NoteSchema(slug="s", content="c", tags=["not valid!"])

    def test_too_many_tags_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            NoteSchema(slug="s", content="c", tags=[f"t{i}" for i in range(11)])

    def test_schema_is_immutable(self) -> None:
        note = NoteSchema(slug="s", content="c")
        with pytest.raises(ValidationError):
            note.slug = "changed"  # type: ignore[misc]


class TestSaveNoteHappyPath:
    """`save_note` writes a well-formed markdown file and returns metadata."""

    def test_writes_file_with_expected_name(
        self, tmp_path: Path, fixed_today: date
    ) -> None:
        note = NoteSchema(slug="standup", content="body")
        result = save_note(note, base_dir=tmp_path, today=fixed_today)

        expected = tmp_path / "2026-07-04-standup.md"
        assert result.path == expected.resolve()
        assert result.filename == "2026-07-04-standup.md"
        assert expected.exists()

    def test_yaml_frontmatter_contains_date_and_slug(
        self, tmp_path: Path, fixed_today: date
    ) -> None:
        note = NoteSchema(slug="notes", content="body")
        result = save_note(note, base_dir=tmp_path, today=fixed_today)
        text = result.path.read_text(encoding="utf-8")

        assert text.startswith("---\n")
        assert "date: 2026-07-04" in text
        assert "slug: notes" in text
        assert text.rstrip().endswith("body")

    def test_tags_render_as_yaml_list(
        self, tmp_path: Path, fixed_today: date
    ) -> None:
        note = NoteSchema(
            slug="tagged", content="body", tags=["one", "two-parts"]
        )
        result = save_note(note, base_dir=tmp_path, today=fixed_today)
        text = result.path.read_text(encoding="utf-8")

        assert "tags:" in text
        assert "  - one" in text
        assert "  - two-parts" in text

    def test_omits_tags_key_when_no_tags(
        self, tmp_path: Path, fixed_today: date
    ) -> None:
        note = NoteSchema(slug="plain", content="body")
        result = save_note(note, base_dir=tmp_path, today=fixed_today)
        text = result.path.read_text(encoding="utf-8")

        assert "tags:" not in text

    def test_creates_missing_directories(
        self, tmp_path: Path, fixed_today: date
    ) -> None:
        nested = tmp_path / "deep" / "nested" / "notes"
        assert not nested.exists()
        note = NoteSchema(slug="s", content="c")
        save_note(note, base_dir=nested, today=fixed_today)
        assert nested.is_dir()

    def test_returns_correct_byte_count(
        self, tmp_path: Path, fixed_today: date
    ) -> None:
        note = NoteSchema(slug="s", content="c")
        result = save_note(note, base_dir=tmp_path, today=fixed_today)
        assert result.bytes_written == result.path.stat().st_size


class TestCollisionSuffix:
    """Existing files must never be silently overwritten."""

    def test_second_save_gets_dash_one_suffix(
        self, tmp_path: Path, fixed_today: date
    ) -> None:
        note = NoteSchema(slug="dup", content="first")
        first = save_note(note, base_dir=tmp_path, today=fixed_today)

        note_two = NoteSchema(slug="dup", content="second")
        second = save_note(note_two, base_dir=tmp_path, today=fixed_today)

        assert first.filename == "2026-07-04-dup.md"
        assert second.filename == "2026-07-04-dup-1.md"
        assert first.path.read_text(encoding="utf-8").rstrip().endswith("first")
        assert second.path.read_text(encoding="utf-8").rstrip().endswith("second")

    def test_multiple_collisions_increment_the_suffix(
        self, tmp_path: Path, fixed_today: date
    ) -> None:
        filenames: list[str] = []
        for _ in range(3):
            note = NoteSchema(slug="dup", content="x")
            filenames.append(save_note(note, base_dir=tmp_path, today=fixed_today).filename)

        assert filenames == [
            "2026-07-04-dup.md",
            "2026-07-04-dup-1.md",
            "2026-07-04-dup-2.md",
        ]


class TestReturnValue:
    """`SaveNoteResult` must be a frozen, well-typed metadata record."""

    def test_result_is_a_pydantic_model(
        self, tmp_path: Path, fixed_today: date
    ) -> None:
        note = NoteSchema(slug="s", content="c")
        result = save_note(note, base_dir=tmp_path, today=fixed_today)
        assert isinstance(result, SaveNoteResult)

    def test_result_is_frozen(
        self, tmp_path: Path, fixed_today: date
    ) -> None:
        note = NoteSchema(slug="s", content="c")
        result = save_note(note, base_dir=tmp_path, today=fixed_today)
        with pytest.raises(ValidationError):
            result.filename = "tampered.md"  # type: ignore[misc]
