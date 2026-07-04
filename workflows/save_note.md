# Workflow: Save Note

## Objective

Capture a short markdown note to the founder's local scratch directory so
that thoughts, meeting recaps, and draft ideas can be persisted without
leaving the assistant conversation. Notes are intentionally disposable and
never leave the workstation; they match the "intermediates local,
deliverables cloud" principle declared in `knowledge/Claude.md`.

## When to Run

Invoke this workflow only when the user has clearly asked to persist text
locally. Typical trigger phrases:

- "save this as a note", "write this down", "capture this"
- "jegyzeteld le", "mentsd el jegyzetnek", "tedd el ezt"

Do **not** invoke it while the user is still thinking out loud. When intent
is ambiguous, ask a single clarifying question before calling the tool.

## Required Inputs

| Field | Type | Rules |
| --- | --- | --- |
| `slug` | `string` | Lowercase kebab-case, 1–60 chars, must match `^[a-z0-9]+(-[a-z0-9]+)*$`. Derive from the note's topic if the user did not supply one. |
| `content` | `string` | Markdown body, 1–20 000 characters. Preserve the user's wording verbatim; do not paraphrase. |
| `tags` | `string[]` | Optional. Each tag must be lowercase kebab-case. Up to 10 tags. |

All three fields are validated by `tools.save_note.NoteSchema` (pydantic)
before the file system is touched, so invalid payloads never reach disk.

## Tool

- **Module**: `tools/save_note.py`
- **Callable**: `save_note(note: NoteSchema) -> SaveNoteResult`
- **Input schema**: `NoteSchema` — expose `NoteSchema.model_json_schema()`
  to the Anthropic tool-use API so the LLM receives the exact contract.
- **Output**: `SaveNoteResult` with `path`, `filename`, and `bytes_written`.

## Expected Output

- **File system**: `.tmp/notes/YYYY-MM-DD-<slug>.md` with a YAML
  frontmatter block (`date`, `slug`, `tags`) followed by the raw markdown
  body. Colliding names automatically receive a `-1`, `-2`, ... suffix, so
  no prior note is ever overwritten.
- **Assistant reply**: a one-line confirmation quoting the resolved
  filename, so the user immediately knows where the note landed.

## Edge Cases

- **Invalid slug** — pydantic raises `ValidationError`. Regenerate a
  compliant slug from the user's topic (strip whitespace, lowercase,
  replace spaces with hyphens) and retry once. If the second attempt still
  fails, surface the error verbatim rather than guessing again.
- **Empty content** — refuse the tool call and ask the user for the note
  body. Never write an empty note.
- **Existing note with the same date and slug** — the tool appends a
  numeric suffix automatically. Report the actual filename in the reply so
  the user is aware the collision occurred.
- **File system error** (permissions, missing disk, read-only mount) —
  `OSError` propagates. Report the error to the user and suggest running
  `mkdir .tmp/notes` manually if the directory is missing.
- **User provides a filename with an extension** — strip the extension and
  pass only the stem to `slug`. The tool always writes `.md`.
- **User provides tags with spaces or punctuation** — normalize to
  lowercase kebab-case before sending; the schema rejects anything else.

## Self-Improvement Notes

Update this workflow (do not silently discard it) whenever you discover a
new failure mode, a better default, or a recurring user pattern. That is
the loop that keeps the WAT framework reliable.

## Related

- Architecture: `knowledge/Claude.md`
- Tool source: `tools/save_note.py`
