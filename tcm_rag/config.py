from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .errors import ConfigurationError


@dataclass(frozen=True)
class IngestConfig:
    pdf_path: str
    page_numbers: Optional[list[int]]
    min_line_length: int
    chunk_size: int
    overlap_size: int


@dataclass(frozen=True)
class EmbeddingsConfig:
    provider: str
    model: str
    dimensions: Optional[int]


@dataclass(frozen=True)
class VectorStoreConfig:
    provider: str
    persist_path: str
    collection_name: str


@dataclass(frozen=True)
class RerankConfig:
    provider: str
    model: str
    enabled: bool


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    enabled: bool


@dataclass(frozen=True)
class AppConfig:
    ingest: IngestConfig
    embeddings: EmbeddingsConfig
    vector_store: VectorStoreConfig
    rerank: RerankConfig
    llm: LLMConfig

    @property
    def repo_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def resolve_path(self, maybe_relative_path: str) -> str:
        p = Path(maybe_relative_path)
        if p.is_absolute():
            return str(p)
        return str((self.repo_root / p).resolve())


def load_config(config_path: str) -> AppConfig:
    path = Path(config_path)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        raise ConfigurationError(f"配置文件不存在: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    return _parse_config(raw)


def _parse_config(raw: dict[str, Any]) -> AppConfig:
    ingest_raw = raw.get("ingest") or {}
    embeddings_raw = raw.get("embeddings") or {}
    vector_store_raw = raw.get("vector_store") or {}
    rerank_raw = raw.get("rerank") or {}
    llm_raw = raw.get("llm") or {}

    ingest = IngestConfig(
        pdf_path=str(ingest_raw.get("pdf_path", "")),
        page_numbers=ingest_raw.get("page_numbers"),
        min_line_length=int(ingest_raw.get("min_line_length", 10)),
        chunk_size=int(ingest_raw.get("chunk_size", 350)),
        overlap_size=int(ingest_raw.get("overlap_size", 120)),
    )
    embeddings = EmbeddingsConfig(
        provider=str(embeddings_raw.get("provider", "openai")),
        model=str(embeddings_raw.get("model", "text-embedding-3-small")),
        dimensions=embeddings_raw.get("dimensions"),
    )
    vector_store = VectorStoreConfig(
        provider=str(vector_store_raw.get("provider", "chroma")),
        persist_path=str(vector_store_raw.get("persist_path", "db/chroma")),
        collection_name=str(vector_store_raw.get("collection_name", "tcm_clinical_assistant")),
    )
    rerank = RerankConfig(
        provider=str(rerank_raw.get("provider", "cross_encoder")),
        model=str(rerank_raw.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")),
        enabled=bool(rerank_raw.get("enabled", False)),
    )
    llm = LLMConfig(
        provider=str(llm_raw.get("provider", "openai")),
        model=str(llm_raw.get("model", "gpt-4o-mini")),
        enabled=bool(llm_raw.get("enabled", False)),
    )

    if not ingest.pdf_path:
        raise ConfigurationError("ingest.pdf_path 不能为空")
    if ingest.min_line_length < 0 or ingest.chunk_size <= 0 or ingest.overlap_size < 0:
        raise ConfigurationError("ingest 参数不合法")

    return AppConfig(
        ingest=ingest,
        embeddings=embeddings,
        vector_store=vector_store,
        rerank=rerank,
        llm=llm,
    )


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return v

