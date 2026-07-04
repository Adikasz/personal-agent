"""Deterministic tool implementations.

WAT layer: **Tool** — every module in this package must be a pure, side-effect-
controlled Python callable. No LLM calls, no reasoning, no hidden global state.
Each tool exposes a pydantic input schema so the Agent layer can safely
translate an LLM-produced JSON payload into a validated function argument
before touching the outside world.
"""
