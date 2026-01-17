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
class ModelScopeConfig:
    cache_dir: str


@dataclass(frozen=True)
class LlamaIndexConfig:
    enabled: bool
    documents_dir: str
    required_exts: list[str]
    persist_dir: str
    chunk_size: int
    similarity_top_k: int
    embedding_model_id: str
    llm_model_id: str
    system_prompt: str
    qa_prompt: str
    refine_prompt: str
    context_window: int
    max_new_tokens: int
    temperature: float
    do_sample: bool
    torch_dtype: str
    device_map: str


@dataclass(frozen=True)
class AppConfig:
    ingest: IngestConfig
    embeddings: EmbeddingsConfig
    vector_store: VectorStoreConfig
    rerank: RerankConfig
    llm: LLMConfig
    modelscope: ModelScopeConfig
    llamaindex: LlamaIndexConfig

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
    modelscope_raw = raw.get("modelscope") or {}
    llamaindex_raw = raw.get("llamaindex") or {}

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

    modelscope = ModelScopeConfig(
        cache_dir=str(modelscope_raw.get("cache_dir", str(Path("data") / "modelscope"))),
    )
    llamaindex = LlamaIndexConfig(
        enabled=bool(llamaindex_raw.get("enabled", False)),
        documents_dir=str(llamaindex_raw.get("documents_dir", "documents")),
        required_exts=list(llamaindex_raw.get("required_exts", [".txt"])),
        persist_dir=str(llamaindex_raw.get("persist_dir", "data/doc_emb")),
        chunk_size=int(llamaindex_raw.get("chunk_size", 256)),
        similarity_top_k=int(llamaindex_raw.get("similarity_top_k", 5)),
        embedding_model_id=str(llamaindex_raw.get("embedding_model_id", "BAAI/bge-base-zh-v1.5")),
        llm_model_id=str(llamaindex_raw.get("llm_model_id", "Qwen/Qwen2.5-7B-Instruct")),
        system_prompt=str(llamaindex_raw.get("system_prompt", "你是一个医疗人工智能助手。")),
        qa_prompt=str(
            llamaindex_raw.get(
                "qa_prompt",
                (
                    "上下文信息如下。\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "请根据上下文信息而不是先验知识来回答以下的查询。作为一个医疗人工智能助手，你的回答要尽可能严谨。\n"
                    "Query: {query_str}\n"
                    "Answer: "
                ),
            )
        ),
        refine_prompt=str(
            llamaindex_raw.get(
                "refine_prompt",
                (
                    "原始查询如下：{query_str}"
                    "我们提供了现有答案：{existing_answer}"
                    "我们有机会通过下面的更多上下文来完善现有答案（仅在需要时）。"
                    "------------"
                    "{context_msg}"
                    "------------"
                    "考虑到新的上下文，优化原始答案以更好地回答查询。 如果上下文没有用，请返回原始答案。"
                    "Refined Answer:"
                ),
            )
        ),
        context_window=int(llamaindex_raw.get("context_window", 4096)),
        max_new_tokens=int(llamaindex_raw.get("max_new_tokens", 2048)),
        temperature=float(llamaindex_raw.get("temperature", 0.0)),
        do_sample=bool(llamaindex_raw.get("do_sample", False)),
        torch_dtype=str(llamaindex_raw.get("torch_dtype", "float16")),
        device_map=str(llamaindex_raw.get("device_map", "auto")),
    )

    if not ingest.pdf_path and not llamaindex.enabled:
        raise ConfigurationError("ingest.pdf_path 不能为空（除非 llamaindex.enabled=true）")
    if ingest.min_line_length < 0 or ingest.chunk_size <= 0 or ingest.overlap_size < 0:
        raise ConfigurationError("ingest 参数不合法")

    return AppConfig(
        ingest=ingest,
        embeddings=embeddings,
        vector_store=vector_store,
        rerank=rerank,
        llm=llm,
        modelscope=modelscope,
        llamaindex=llamaindex,
    )


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return v
