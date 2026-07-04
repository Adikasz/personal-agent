# Workflow: Search Notes

## Objective

Retrieve previously saved markdown notes so the assistant can answer
recall questions ("what did I write about hiring last week?", "find my
Q3 planning notes") without the user having to open a file manager.
Together with `save_note`, this closes the save-and-retrieve loop on
the local scratchpad.

## When to Run

Invoke this workflow when the user asks to find, look up, recall, or
reference something they saved earlier. Typical trigger phrases:

- "find my notes on…", "did I write anything about…", "look up…"
- "keresd meg a jegyzeteket…", "mit írtam a…", "nézd meg…"

Do **not** invoke it for questions the user could obviously answer from
their own conversation memory in the current session. When intent is
ambiguous, ask a single clarifying question before calling the tool.

## Required Inputs

| Field | Type | Rules |
| --- | --- | --- |
| `query` | `string` | 1–200 characters, non-whitespace. Case-insensitive substring match against the note body. |
| `tags` | `string[]` | Optional. Each tag lowercase kebab-case, up to 10. When present, only notes carrying **every** listed tag are returned. |
| `limit` | `int` | Optional (default 5, 1–20). Upper bound on the number of matches. |

All fields are validated by `tools.search_notes.SearchNotesQuery`
(pydantic) before the file system is touched.

## Tool

- **Module**: `tools/search_notes.py`
- **Callable**: `search_notes(query: SearchNotesQuery) -> SearchNotesResult`
- **Input schema**: `SearchNotesQuery` — expose
  `SearchNotesQuery.model_json_schema()` to the Anthropic tool-use API.
- **Output**: `SearchNotesResult` with `query`, `scanned`, and a list of
  `NoteMatch` (`filename`, `date`, `slug`, `tags`, `snippet`).

## Expected Output

- **Assistant reply**: at most a handful of matches. For each match,
  quote the filename and the snippet, then offer to elaborate. Do **not**
  fabricate content that is not present in the snippet — if the user
  wants more, ask before opening the file.
- **File system**: no writes; this workflow is read-only.

## Edge Cases

- **Zero matches** — say so plainly, echo the query back, and suggest
  the user broaden the search or check a different tag.
- **Directory missing** — the tool returns an empty result with
  `scanned=0`. Tell the user they have not saved any notes yet.
- **Overly broad query** — if the user asked with a single common word
  and hundreds of notes match, the tool truncates at `limit`. Report the
  truncation and offer to narrow the query.
- **Query with special characters** — the tool matches literally
  (case-insensitive substring), not regex. Do not escape the query on
  the user's behalf.
- **Third-party file without frontmatter** — the tool still searches
  the body but returns empty `date`, `slug`, and `tags`. Quote the
  filename verbatim rather than inventing metadata.

## Self-Improvement Notes

Refine this workflow when you notice recurring miss patterns (for
example, users routinely searching for topics that are tagged
inconsistently). Prefer proposing a new tag convention over silently
expanding the tool's surface area.

## Related

- Companion tool: `workflows/save_note.md`
- Architecture: `knowledge/Claude.md`
- Tool source: `tools/search_notes.py`
