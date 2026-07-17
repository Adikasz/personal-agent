"""Application configuration loaded from environment variables.

WAT layer: **Tool** — deterministic infrastructure with no LLM calls and no
reasoning. All secrets and runtime knobs are declared here as a single, typed
source of truth. The `Settings` object is cached via `get_settings()` so the
`.env` file is parsed exactly once per process.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Final

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Strictly typed application settings sourced from `.env`."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    anthropic_api_key: SecretStr = Field(
        ...,
        description="API key used to authenticate against the Anthropic API.",
    )
    anthropic_model: str = Field(
        default="claude-opus-4-7",
        description="Default Anthropic model identifier.",
    )

    openai_api_key: SecretStr = Field(
        ...,
        description="API key used to authenticate against the OpenAI API for embeddings.",
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model identifier used by the RAG store.",
    )

    pinecone_api_key: SecretStr = Field(
        ...,
        description="API key used to authenticate against Pinecone.",
    )
    pinecone_index_name: str = Field(
        default="plansmart",
        description="Pinecone index name used for vector storage.",
    )

    tavily_api_key: SecretStr = Field(
        ...,
        description=(
            "API key used to authenticate against Tavily. Injected into the "
            "environment of the stdio MCP server subprocess so the "
            "`tavily_search` tool can reach the Tavily API."
        ),
    )
    anthropic_max_tokens: int = Field(
        default=2048,
        gt=0,
        le=8192,
        description="Maximum number of tokens the model may return per call.",
    )

    max_history_turns: int = Field(
        default=20,
        gt=0,
        le=200,
        description=(
            "Maximum number of messages retained in the assistant's "
            "conversation history. Older messages are dropped in FIFO order."
        ),
    )
    max_tool_iterations: int = Field(
        default=5,
        gt=0,
        le=20,
        description=(
            "Upper bound on the number of tool-use rounds the assistant "
            "may perform within a single `ask()` invocation before it is "
            "forced to return a final answer to the user."
        ),
    )

    log_level: str = Field(
        default="INFO",
        description="Logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    knowledge_dir: Path = Field(
        default=PROJECT_ROOT / "knowledge",
        description="Directory containing the agent's knowledge base.",
    )

    @property
    def resolved_knowledge_dir(self) -> Path:
        """Return the knowledge directory as an absolute path."""
        path = self.knowledge_dir
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached, singleton `Settings` instance."""
    return Settings()  # type: ignore[call-arg]
