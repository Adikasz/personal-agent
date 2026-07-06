"""Unit tests for `tools.search_notes`.

Every test writes its fixture notes to `tmp_path` so nothing ever
touches the real `.tmp/notes/` directory. The notes are produced by
`save_note` where possible, which keeps the frontmatter shape aligned
with production.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from tools.save_note import NoteSchema, save_note
from tools.search_notes import (
    NoteMatch,
    SearchNotesQuery,
    SearchNotesResult,
    search_notes,
)


@pytest.fixture
def frozen_today() -> date:
    return date(2026, 7, 4)


@pytest.fixture
def populated_notes(tmp_path: Path, frozen_today: date) -> Path:
    """Seed `tmp_path` with three heterogeneous notes and return it."""
    save_note(
        NoteSchema(
            slug="q3-planning",
            content="Roadmap items for Q3 include pricing and hiring.",
            tags=["planning", "q3"],
        ),
        base_dir=tmp_path,
        today=frozen_today,
    )
    save_note(
        NoteSchema(
            slug="standup",
            content="Discussed hiring pipeline and interview scorecards.",
            tags=["hiring"],
        ),
        base_dir=tmp_path,
        today=frozen_today,
    )
    save_note(
        NoteSchema(
            slug="grocery",
            content="Buy milk, eggs, and coffee beans.",
            tags=["personal"],
        ),
        base_dir=tmp_path,
        today=frozen_today,
    )
    return tmp_path


class TestQueryValidation:
    """`SearchNotesQuery` mirrors the strict contract of `NoteSchema`."""

    def test_valid_query_round_trips(self) -> None:
        q = SearchNotesQuery(query="hiring", tags=["q3"], limit=3)
        assert q.query == "hiring"
        assert q.tags == ["q3"]
        assert q.limit == 3

    def test_default_tags_and_limit(self) -> None:
        q = SearchNotesQuery(query="anything")
        assert q.tags == []
        assert q.limit == 5

    def test_empty_query_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SearchNotesQuery(query="")

    def test_whitespace_only_query_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SearchNotesQuery(query="   ")

    def test_overly_long_query_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SearchNotesQuery(query="x" * 201)

    def test_tag_with_invalid_characters_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SearchNotesQuery(query="hiring", tags=["not valid!"])

    def test_too_many_tags_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SearchNotesQuery(query="q", tags=[f"t{i}" for i in range(11)])

    def test_limit_bounds_are_enforced(self) -> None:
        with pytest.raises(ValidationError):
            SearchNotesQuery(query="q", limit=0)
        with pytest.raises(ValidationError):
            SearchNotesQuery(query="q", limit=21)

    def test_extra_fields_are_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            SearchNotesQuery(query="q", nonsense="drop me")  # type: ignore[call-arg]

    def test_query_is_immutable(self) -> None:
        q = SearchNotesQuery(query="hiring")
        with pytest.raises(ValidationError):
            q.query = "changed"  # type: ignore[misc]


class TestSearchBehaviour:
    """`search_notes` returns structured, well-formed matches."""

    def test_missing_directory_returns_empty_result(self, tmp_path: Path) -> None:
        result = search_notes(
            SearchNotesQuery(query="anything"),
            base_dir=tmp_path / "does-not-exist",
        )
        assert isinstance(result, SearchNotesResult)
        assert result.matches == []
        assert result.scanned == 0
        assert result.query == "anything"

    def test_matching_query_returns_expected_note(self, populated_notes: Path) -> None:
        result = search_notes(
            SearchNotesQuery(query="roadmap"),
            base_dir=populated_notes,
        )
        assert result.scanned == 3
        assert len(result.matches) == 1
        match = result.matches[0]
        assert isinstance(match, NoteMatch)
        assert match.slug == "q3-planning"
        assert match.date == "2026-07-04"
        assert "planning" in match.tags
        assert "roadmap" in match.snippet.lower()

    def test_case_insensitive_match(self, populated_notes: Path) -> None:
        result = search_notes(
            SearchNotesQuery(query="HIRING"),
            base_dir=populated_notes,
        )
        slugs = [match.slug for match in result.matches]
        assert "standup" in slugs

    def test_no_match_returns_empty_but_still_reports_scanned(self, populated_notes: Path) -> None:
        result = search_notes(
            SearchNotesQuery(query="blockchain"),
            base_dir=populated_notes,
        )
        assert result.matches == []
        assert result.scanned == 3

    def test_tag_filter_restricts_matches(self, populated_notes: Path) -> None:
        result = search_notes(
            SearchNotesQuery(query="hiring", tags=["hiring"]),
            base_dir=populated_notes,
        )
        assert [match.slug for match in result.matches] == ["standup"]

    def test_tag_filter_requires_all_tags(self, populated_notes: Path) -> None:
        # Only q3-planning is tagged with both "planning" AND "q3".
        result = search_notes(
            SearchNotesQuery(query="pricing", tags=["planning", "q3"]),
            base_dir=populated_notes,
        )
        assert [match.slug for match in result.matches] == ["q3-planning"]

        # No note has both "hiring" AND "planning".
        result = search_notes(
            SearchNotesQuery(query="hiring", tags=["hiring", "planning"]),
            base_dir=populated_notes,
        )
        assert result.matches == []

    def test_limit_truncates_matches(self, tmp_path: Path, frozen_today: date) -> None:
        for index in range(5):
            save_note(
                NoteSchema(
                    slug=f"item-{index}",
                    content=f"shared keyword and unique body {index}",
                ),
                base_dir=tmp_path,
                today=frozen_today,
            )
        result = search_notes(
            SearchNotesQuery(query="shared keyword", limit=2),
            base_dir=tmp_path,
        )
        assert len(result.matches) == 2
        assert result.scanned == 2  # short-circuits once the cap is hit

    def test_snippet_is_bounded_and_contains_the_match(
        self, tmp_path: Path, frozen_today: date
    ) -> None:
        long_body = "leading padding. " * 20 + "MATCH_ANCHOR" + " trailing padding." * 20
        save_note(
            NoteSchema(slug="long", content=long_body),
            base_dir=tmp_path,
            today=frozen_today,
        )
        result = search_notes(
            SearchNotesQuery(query="match_anchor"),
            base_dir=tmp_path,
        )
        assert len(result.matches) == 1
        snippet = result.matches[0].snippet
        assert "MATCH_ANCHOR" in snippet
        # 80 chars either side plus optional ellipses — well under 300.
        assert len(snippet) <= 300

    def test_unreadable_file_is_skipped_gracefully(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frozen_today: date
    ) -> None:
        save_note(
            NoteSchema(slug="ok", content="findable content"),
            base_dir=tmp_path,
            today=frozen_today,
        )
        broken = tmp_path / "2026-07-04-broken.md"
        broken.write_text("", encoding="utf-8")

        original_read_text = Path.read_text

        def _read_text(self: Path, *args: object, **kwargs: object) -> str:
            if self.name == "2026-07-04-broken.md":
                raise OSError("permission denied")
            return original_read_text(self, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(Path, "read_text", _read_text)

        result = search_notes(
            SearchNotesQuery(query="findable"),
            base_dir=tmp_path,
        )
        assert result.scanned == 2
        assert [match.slug for match in result.matches] == ["ok"]

    def test_third_party_file_without_frontmatter_is_still_searchable(self, tmp_path: Path) -> None:
        (tmp_path / "2026-07-04-external.md").write_text(
            "no frontmatter here, but the word roadmap still appears.",
            encoding="utf-8",
        )
        result = search_notes(
            SearchNotesQuery(query="roadmap"),
            base_dir=tmp_path,
        )
        assert len(result.matches) == 1
        assert result.matches[0].filename == "2026-07-04-external.md"
        assert result.matches[0].tags == []
