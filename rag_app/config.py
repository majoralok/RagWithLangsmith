"""Environment-backed application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class ConfigurationError(ValueError):
    """Raised when application configuration is invalid or incomplete."""


def _integer(name: str, default: int, minimum: int = 1) -> int:
    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer, got {raw_value!r}.") from exc
    if value < minimum:
        raise ConfigurationError(f"{name} must be at least {minimum}.")
    return value


def _float(name: str, default: float) -> float:
    raw_value = os.getenv(name, str(default))
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number, got {raw_value!r}.") from exc


def _is_true(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class AppSettings:
    hf_token: str
    hf_model: str
    hf_provider: str
    chroma_path: Path
    collection_name: str
    top_k: int
    min_relevance_score: float
    chunk_size: int
    chunk_overlap: int
    max_new_tokens: int
    llm_temperature: float
    langsmith_tracing_enabled: bool

    @classmethod
    def from_environment(cls) -> "AppSettings":
        chunk_size = _integer("CHUNK_SIZE", 1000, minimum=100)
        chunk_overlap = _integer("CHUNK_OVERLAP", 150, minimum=0)
        if chunk_overlap >= chunk_size:
            raise ConfigurationError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

        min_score = _float("MIN_RELEVANCE_SCORE", 0.0)
        if not -1.0 <= min_score <= 1.0:
            raise ConfigurationError(
                "MIN_RELEVANCE_SCORE must be between -1.0 and 1.0."
            )

        temperature = _float("LLM_TEMPERATURE", 0.1)
        if not 0.0 <= temperature <= 2.0:
            raise ConfigurationError("LLM_TEMPERATURE must be between 0.0 and 2.0.")

        return cls(
            hf_token=os.getenv("HF_TOKEN", "").strip(),
            hf_model=os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct").strip(),
            hf_provider=os.getenv("HF_PROVIDER", "auto").strip(),
            chroma_path=Path(os.getenv("CHROMA_PATH", ".rag_index")),
            collection_name=os.getenv("COLLECTION_NAME", "knowledge_base").strip(),
            top_k=_integer("TOP_K", 4),
            min_relevance_score=min_score,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_new_tokens=_integer("MAX_NEW_TOKENS", 512),
            llm_temperature=temperature,
            langsmith_tracing_enabled=_is_true(os.getenv("LANGSMITH_TRACING")),
        )

    def require_hugging_face_token(self) -> None:
        if not self.hf_token:
            raise ConfigurationError(
                "HF_TOKEN is missing. Copy .env.example to .env and add your token."
            )

