# Workflow: Morning Briefing

## Objective

Give the founder a focused daily kick-off: surface what he wrote down
recently, distill it into the two or three priorities that actually
matter today, and offer to save the synthesis as a fresh digest note.
The workflow is the first proof that the assistant can chain tools
under its own reasoning without any Python-level orchestration.

## When to Run

Trigger this workflow when the user says any of:

- "give me a morning briefing", "start my day", "what's on today"
- "reggeli briefing", "kezdjük a napot", "mi van ma"

Do **not** invoke it if the user is clearly asking a specific question
that a single `search_notes` call can answer. This SOP earns its cost
only when a synthesis step is required.

## Sequence of Steps

1. **Gather** — Call `search_notes` at least once. Start with a broad
   query the user likely wrote about recently ("plans", "priorities",
   "todo", or their previous briefing's tags). If the first call
   returns fewer than three matches, run one more call with a
   complementary query rather than fabricating context.
2. **Consult, if needed** — If the search returns notes whose snippets
   are ambiguous, use `read_file` to open the specific file for full
   context. Never guess at content that is not visible in a snippet.
3. **Synthesize** — Produce a brief that is at most:
   - 1–2 sentences of situational summary, and
   - 2–3 numbered priorities, each with a one-line rationale grounded
     in the notes you actually retrieved. Quote filenames as evidence
     when a priority derives from a specific note.
4. **Present** — Reply to the user in the language they used. Keep the
   tone concise, direct, and enterprise-grade (matching the assistant's
   default voice).
5. **Offer to persist** — End the reply with a single-line question:
   "Save this briefing as `YYYY-MM-DD-briefing`?". Do **not** call
   `save_note` speculatively. Wait for confirmation, then invoke
   `save_note` with a clean kebab-case slug and the digest text as
   `content`. Tag the note with `briefing` and any topical tag that
   dominated the synthesis.

## Guard Rails

- If no notes match any reasonable query, say so plainly. Offer to
  start the day with an intake question ("what's the biggest thing on
  your mind?") rather than inventing priorities.
- Never open files outside the project workspace. The `read_file` and
  `list_directory` tools already refuse escape attempts with
  `SecurityError`; do not try to work around that boundary.
- Never overwrite an existing briefing. `save_note` will suffix
  collisions automatically; report the exact filename in the reply.

## Related

- Companion workflows: `workflows/save_note.md`,
  `workflows/search_notes.md`.
- Architecture: `knowledge/Claude.md`.
- Tools invoked: `tools/search_notes.py`, `tools/read_file.py`,
  `tools/save_note.py`; `tools/list_directory.py` is also available if
  you want to browse the workspace before searching.
