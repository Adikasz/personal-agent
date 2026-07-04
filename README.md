<div align="center">

# PlanSmart — WAT Career & Operations Agent

**An autonomous CLI strategist that turns a founder's day into a compounding portfolio artifact.**

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Anthropic Claude](https://img.shields.io/badge/LLM-Claude%20Opus%204-CC785C)](https://www.anthropic.com/)
[![Pinecone RAG](https://img.shields.io/badge/RAG-Pinecone-1E1E1E)](https://www.pinecone.io/)
[![OpenAI Embeddings](https://img.shields.io/badge/Embeddings-OpenAI-412991?logo=openai&logoColor=white)](https://platform.openai.com/)
[![AsyncIO](https://img.shields.io/badge/Runtime-AsyncIO-orange)](https://docs.python.org/3/library/asyncio.html)
[![Pydantic](https://img.shields.io/badge/Contracts-Pydantic%20v2-E92063)](https://docs.pydantic.dev/)
[![Tests](https://img.shields.io/badge/Tests-237%20unit%20%2B%20integration-2ea44f)](#quality-bar)
[![Type-checked](https://img.shields.io/badge/mypy-strict-blue)](https://mypy.readthedocs.io/)
[![Lint](https://img.shields.io/badge/ruff-clean-46A3FF)](https://docs.astral.sh/ruff/)
[![License](https://img.shields.io/badge/License-Proprietary-lightgrey)](#license)

</div>

---

## 1. Executive Summary

**PlanSmart** is a production-grade, single-operator CLI agent that acts as a Personal Career & Operations Strategist for a founder pursuing $50+/hr remote AI Engineer roles and scaling an automation agency.

It is **not** a chatbot wrapped around an LLM. It is a **deterministic reasoning system** built on the **WAT (Workflows, Agents, Tools)** framework, with hard boundaries between the layer that *decides* (Agent), the layer that *acts* (Tools), and the layer that *governs* (Workflows / SOPs).

The agent chains three institutional capabilities on every turn:

| Capability | Concrete Effect |
| --- | --- |
| **Deterministic Workflows** | Codified SOPs for career-critical tasks (job analysis, morning briefing, note recall). No prompt drift. |
| **Asynchronous Web Senses** | Live DuckDuckGo search + clean URL scraping, dispatched off the event loop. |
| **Enterprise Memory (RAG)** | OpenAI-embedded, Pinecone-persisted long-term context. Semantic recall over prior work, résumé fragments, and researched articles. |

### The Business Case

| Metric | Before | With PlanSmart |
| --- | --- | --- |
| Time to analyze + prep a job posting | 45–90 min | **< 5 min**, structured output |
| Résumé/experience recall | Manual document hunting | **Semantic vector search** |
| Note capture + retrieval | Scattered across apps | Local + indexed to RAG in one command |
| Application quality | Generic | Custom **"Hybrid Advantage" pitch** per role |
| Repository defensibility | Unproven | **237 tests, strict typing, zero-network CI** |

**ROI hypothesis:** one well-targeted application per week that PlanSmart would not otherwise have produced is worth a $10K+/yr salary delta at the founder's rate band. The system pays for itself before the first offer.

---

## 2. The WAT Architecture

PlanSmart enforces a strict, three-layer separation. Each layer has one job and cannot cross-cut into the others.

```
┌───────────────────────────────────────────────────────────────────────────┐
│                          WORKFLOWS  (workflows/*.md)                      │
│  Deterministic SOPs the Agent must read + follow verbatim before acting.  │
│    • job_application_prep.md   • morning_briefing.md                      │
│    • save_note.md              • search_notes.md                          │
└───────────────────────────▲───────────────────────────────────────────────┘
                            │ read_file / list_directory
┌───────────────────────────┴───────────────────────────────────────────────┐
│                          AGENT  (agents/personal_assistant.py)            │
│  Bounded tool-use loop over Claude Opus 4. Grounded system prompt.        │
│  Pydantic-validated tool I/O. Errors surfaced to the model, never raised. │
│  History-safe (tool_use / tool_result pairs never split on trim).         │
└───────────────────────────▲───────────────────────────────────────────────┘
                            │ typed tool calls (JSON Schema)
┌───────────────────────────┴───────────────────────────────────────────────┐
│                          TOOLS  (tools/*.py)                              │
│  ┌─── Local sandboxed I/O ────┐ ┌──── Async Web Senses ────┐ ┌── RAG ──┐  │
│  │ save_note                  │ │ web_search  (DuckDuckGo) │ │ semantic │  │
│  │ search_notes               │ │ scrape_url  (httpx+BS4)  │ │ _search  │  │
│  │ list_directory / read_file │ └──────────────────────────┘ │ index_   │  │
│  └────────────────────────────┘                              │ document │  │
│                                                              └──────────┘  │
└───────────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │ Enterprise Memory     │
                │ Pinecone + OpenAI     │
                │ text-embedding-3-small│
                └───────────────────────┘
```

### Why this matters at the CTO level

- **Auditability.** Every capability is a Pydantic-typed function with a JSON Schema published to the LLM. A hallucinated tool call is a validation error, not a runtime crash.
- **Vendor swap safety.** `utils/vector_store.py` is the **only** module in the codebase that imports `openai` or `pinecone`. Change vendors in one file, not fifty.
- **Non-blocking I/O.** Every network-bound tool (DuckDuckGo, httpx, Pinecone) is dispatched via `asyncio.to_thread` or a native async client — the reasoning loop never stalls.
- **Bounded loops.** `max_tool_iterations` caps any single turn. No runaway agents, no runaway bills.

---

## 3. The Agent Loop, End-to-End

```
 user > analyze this job: https://example.com/roles/ai-engineer

  ├─▶ [agent] calling tool: list_directory(path=workflows)
  │           ↳ returns job_application_prep.md, morning_briefing.md, …
  ├─▶ [agent] calling tool: read_file(path=workflows/job_application_prep.md)
  │           ↳ SOP loaded, steps followed verbatim
  ├─▶ [agent] calling tool: scrape_url(url=…)              ← Web Sense
  ├─▶ [agent] calling tool: semantic_search(query=…)       ← Enterprise Memory
  ├─▶ [agent] calling tool: web_search(query="<company> engineering blog")
  │
  └─▶ agent > Match Score: 82% ▸ Hybrid Advantage pitch ▸ Skill Gaps ▸ Cover Letter Hook
```

Each `[agent]` line is a real, typed, validated tool invocation — printed live so the operator can audit reasoning in real time.

---

## 4. Feature Matrix

### 4.1 Deterministic Workflows

Machine-readable SOPs in `workflows/`. The Agent is instructed to discover and follow them before improvising.

| Workflow | Trigger | Outcome |
| --- | --- | --- |
| **`job_application_prep.md`** | "analyze this job", URL provided | Match score, Hybrid Advantage pitch, skill gaps, cover-letter hook |
| **`morning_briefing.md`** | "morning briefing", "start my day" | Priority triage across recent notes + synthesis digest |
| **`save_note.md`** | "save this", "write this down" | Timestamped markdown persisted locally |
| **`search_notes.md`** | "find my notes on…" | Keyword + tag search with snippet windows |

Adding a new capability is usually a `.md` file, not a code deploy.

### 4.2 Asynchronous Web Senses

| Tool | Backing | Failure Mode |
| --- | --- | --- |
| `web_search` | `duckduckgo-search` via `asyncio.to_thread` | Returns typed result with `error` field; loop keeps running |
| `scrape_url` | `httpx` (async) + `BeautifulSoup` cleanup | Same graceful degradation contract |

Network failures **never** crash the agent. They surface to the LLM as structured `is_error=True` tool results, and the model self-corrects on the next iteration.

### 4.3 Enterprise Memory (Pinecone RAG)

| Tool | What it does |
| --- | --- |
| `index_document` | Embeds text via OpenAI `text-embedding-3-small`, upserts to Pinecone with metadata (source URL, filepath, tags) |
| `semantic_search` | Meaning-based recall over indexed résumé fragments, scraped research, and long-form notes |

The vector store is fronted by a single async facade (`utils/vector_store.py`) with its own `VectorStoreError` type, so downstream tools never leak vendor exceptions.

### 4.4 Sandboxed Local I/O

`utils/sandbox.py` guarantees every path passed to `read_file`, `list_directory`, `save_note`, or `search_notes` is resolved **inside** the project root. Absolute paths and traversal attempts (`../../etc/…`) are refused before the disk is touched.

---

## 5. Quality Bar

| Discipline | Standard |
| --- | --- |
| **Unit tests** | **200+ zero-network tests** across `tests/unit/` covering every tool, the agent loop, the vector-store facade, sandboxing, history trimming, and config validation. Vendor SDKs are mocked at the seam. |
| **Integration tests** | End-to-end workflow execution scenarios in `tests/integration/`. |
| **Static typing** | `mypy --strict` across the codebase. No implicit `Any` in first-party code. |
| **Linting** | `ruff` with `E, W, F, I, B, UP, C4, SIM, RUF` rulesets enabled. |
| **Warnings-as-errors** | `pytest` runs with `filterwarnings = ["error"]`. Deprecations become failures. |
| **Config isolation** | Every secret loaded via `pydantic-settings` from `.env`. Missing keys fail loudly at import time, not silently at first use. |
| **Logging** | Structured `logging` throughout. `print` is banned in production paths. |

Run the full pipeline locally:

```bash
make ci        # ruff + mypy strict + pytest
```

---

## 6. Tech Stack

| Layer | Choice | Rationale |
| --- | --- | --- |
| **LLM** | Anthropic Claude Opus 4 (`claude-opus-4-7`) | Best-in-class tool use, long context, deterministic reasoning |
| **Embeddings** | OpenAI `text-embedding-3-small` | Cost/latency floor for semantic recall |
| **Vector DB** | Pinecone (5.x SDK) | Managed, low-ops, region-flexible |
| **Config** | `pydantic-settings` v2 | Typed secrets, `.env` isolation, `SecretStr` for keys |
| **HTTP** | `httpx` (async) | Modern async client with connection pooling |
| **Search** | `duckduckgo-search` | No API key, no ToS drama, good enough for research |
| **Runtime** | `asyncio` | Non-blocking tool dispatch, ready for parallel tools |
| **Testing** | `pytest` + `pytest-asyncio` (auto mode) | First-class async test support |
| **Lint / Format** | `ruff` | Single binary, sub-second feedback |
| **Type Check** | `mypy` strict | Contract enforcement at CI |

---

## 7. Repository Layout

```
plansmart-dev-workspace/
├─ agents/                      # WAT: Agent — reasoning + tool-use loop
│  └─ personal_assistant.py
├─ tools/                       # WAT: Tools — deterministic capabilities
│  ├─ web_search.py             # DuckDuckGo (async-wrapped)
│  ├─ scrape_url.py             # httpx + BeautifulSoup
│  ├─ semantic_search.py        # Pinecone recall
│  ├─ index_document.py         # OpenAI embed → Pinecone upsert
│  ├─ save_note.py / search_notes.py
│  └─ list_directory.py / read_file.py
├─ workflows/                   # WAT: Workflows — SOPs the Agent must follow
│  ├─ job_application_prep.md
│  ├─ morning_briefing.md
│  ├─ save_note.md
│  └─ search_notes.md
├─ core/
│  └─ config.py                 # Typed settings, secret isolation
├─ utils/
│  ├─ vector_store.py           # Sole Pinecone + OpenAI facade
│  ├─ history_manager.py        # tool_use/tool_result-safe trimming
│  ├─ sandbox.py                # Path-traversal defense
│  └─ logger.py                 # Structured logging setup
├─ knowledge/                   # System-prompt-injected persona + context
├─ tests/
│  ├─ unit/                     # 200+ zero-network tests
│  └─ integration/              # Full workflow execution
├─ scripts/dev.ps1              # Windows dev helper
├─ main.py                      # Async REPL entry point
├─ Makefile                     # install / lint / typecheck / test / ci
├─ pyproject.toml               # ruff + mypy + pytest config
├─ requirements.txt             # Runtime deps
├─ requirements-dev.txt         # Dev deps
└─ env.example                  # Rename to `.env` locally
```

---

## 8. Installation

### Prerequisites

- Python **3.11+**
- API keys for Anthropic, OpenAI, and Pinecone
- (Optional) `make` — bundled on macOS/Linux, install via Git Bash or Chocolatey on Windows

### Setup

```bash
# 1. Clone
git clone <repo-url> plansmart && cd plansmart

# 2. Create + activate a virtualenv
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install
make install-dev          # or:  pip install -r requirements-dev.txt

# 4. Configure secrets
cp env.example .env       # then fill in the three API keys

# 5. Sanity-check the pipeline
make ci
```

### Launch the REPL

```bash
python main.py
```

```
[system] PlanSmart assistant online. /reset to clear history, empty line to quit.
you > analyze this job: https://example.com/careers/ai-engineer
```

Commands inside the REPL:

| Input | Effect |
| --- | --- |
| `/reset` or `/clear` | Discard conversation history without restarting the process |
| *empty line* | Exit |
| `Ctrl-C` / `Ctrl-D` | Exit |

---

## 9. Configuration Reference

All settings are loaded via `pydantic-settings` from `.env`. See `env.example` for the full template.

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `ANTHROPIC_API_KEY` | ✅ | — | Claude authentication |
| `OPENAI_API_KEY` | ✅ | — | Embedding authentication |
| `PINECONE_API_KEY` | ✅ | — | Vector store authentication |
| `ANTHROPIC_MODEL` | ⛔ | `claude-opus-4-7` | Reasoning model |
| `OPENAI_EMBEDDING_MODEL` | ⛔ | `text-embedding-3-small` | Embedding model |
| `PINECONE_INDEX_NAME` | ⛔ | `plansmart` | Target Pinecone index |
| `ANTHROPIC_MAX_TOKENS` | ⛔ | `2048` | Per-call ceiling |
| `MAX_HISTORY_TURNS` | ⛔ | `20` | Conversation memory depth |
| `MAX_TOOL_ITERATIONS` | ⛔ | `5` | Tool-use rounds per user turn |
| `LOG_LEVEL` | ⛔ | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `KNOWLEDGE_DIR` | ⛔ | `knowledge` | System-prompt grounding directory |

**Missing required keys fail at import time**, not silently at first use — a deliberate design choice.

---

## 10. Extending PlanSmart

### Add a new Tool

1. Create `tools/my_tool.py` with a Pydantic `Query` schema and a callable.
2. Register a `ToolSpec` inside `_default_tools()` in `agents/personal_assistant.py`.
3. Add a `tests/unit/test_my_tool.py` with mocked I/O — zero network required.

### Add a new Workflow

1. Drop a new `workflows/my_workflow.md` describing objective, trigger phrases, and step sequence.
2. That's it. The Agent discovers it via `list_directory` and follows it via `read_file`.

### MCP Readiness

The tool registry is intentionally schema-first: every tool ships with a Pydantic `input_schema` that is already JSON-Schema-compatible with the **Model Context Protocol**. Wrapping the current registry as an MCP server is a mechanical translation, not a rewrite.

---

## 11. Design Principles

- **Deterministic over clever.** Workflows are Markdown SOPs, not prompt strings hidden in code.
- **Errors are values.** Every tool has an `error` field; nothing propagates.
- **One place for one dependency.** OpenAI + Pinecone live behind a single facade.
- **Zero-network tests.** Reliability is proven without paying an API bill.
- **Secrets never touch source.** `SecretStr` + `.env` + `.gitignore`. Non-negotiable.
- **Async at every network boundary.** The reasoning loop is never blocked.
- **README-driven development.** If it isn't documented here, it isn't shipped.

---

## 12. Roadmap

- [ ] MCP server wrapper around the existing tool registry
- [ ] LinkedIn outreach workflow (draft + tone-check + send-queue)
- [ ] Automated résumé indexer (folder watcher → `index_document`)
- [ ] Streaming responses in the REPL
- [ ] Multi-user isolation (per-user Pinecone namespace)
- [ ] Prometheus metrics for tool call counts, latencies, and error rates

---

## 13. About the Author

Built by **Dávid** — founder of **PlanSmart**, an AI automation agency that translates business pain points (ROI, ops bottlenecks, process cost) into agentic AI systems.

**The Hybrid Advantage:** BGE (Budapest Business University) International Business & Business Intelligence graduate, 4.84 GPA, C1 English, currently completing the **Anthropic Architect** track. Not "an engineer who learned business" — a business strategist who ships production Python.

Targeting remote **AI Product Engineer** / **AI Solutions Architect** roles with US and EU teams that value both sides of that ledger.

---

## License

Proprietary. All rights reserved. Contact the author for commercial use inquiries.
