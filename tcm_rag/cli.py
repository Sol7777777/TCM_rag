from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import AppConfig, load_config
from .embeddings.openai_embedder import OpenAIEmbedder
from .embeddings.sbert import SentenceTransformerEmbedder
from .ingest.pdf import extract_paragraphs_from_pdf
from .pipeline.rag import RAGPipeline
from .rerank.cross_encoder import CrossEncoderReranker
from .text.splitter import split_text
from .vectorstores.chroma_store import ChromaVectorStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tcm-rag")
    parser.add_argument(
        "--config",
        default=str(Path("configs") / "app.json"),
        help="配置文件路径（JSON）",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="解析PDF并写入向量库")
    p_ingest.add_argument("--reset", action="store_true", help="重置向量库（仅部分实现支持）")

    p_query = sub.add_parser("query", help="检索问答")
    p_query.add_argument("text", help="用户问题")
    p_query.add_argument("--top-k", type=int, default=5)
    p_query.add_argument("--print-prompt", action="store_true")
    p_query.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.command == "ingest":
        _cmd_ingest(cfg, reset=bool(args.reset))
        return 0
    if args.command == "query":
        _cmd_query(cfg, query=args.text, top_k=int(args.top_k), print_prompt=bool(args.print_prompt), as_json=bool(args.json))
        return 0

    parser.print_help()
    return 2


def _cmd_ingest(cfg: AppConfig, *, reset: bool) -> None:
    pdf_path = cfg.resolve_path(cfg.ingest.pdf_path)
    paragraphs = extract_paragraphs_from_pdf(
        pdf_path,
        page_numbers=cfg.ingest.page_numbers,
        min_line_length=cfg.ingest.min_line_length,
    )
    chunks = split_text(
        paragraphs,
        chunk_size=cfg.ingest.chunk_size,
        overlap_size=cfg.ingest.overlap_size,
    )

    embedder = _build_embedder(cfg)
    vectors = embedder.embed(chunks)

    store = _build_store(cfg, reset=reset)
    store.add(
        ids=[f"ck_{i}" for i in range(len(chunks))],
        embeddings=vectors,
        documents=chunks,
        metadatas=[{"source": cfg.ingest.pdf_path, "chunk_index": i} for i in range(len(chunks))],
    )
    print(f"已写入向量库: {len(chunks)} chunks")


def _cmd_query(
    cfg: AppConfig,
    *,
    query: str,
    top_k: int,
    print_prompt: bool,
    as_json: bool,
) -> None:
    embedder = _build_embedder(cfg)
    store = _build_store(cfg, reset=False)
    reranker = _build_reranker(cfg)

    pipe = RAGPipeline(vector_store=store, embedder=embedder, reranker=reranker, llm=None)
    result = pipe.query(user_query=query, top_k=top_k)

    if as_json:
        payload = {
            "query": result.query,
            "contexts": result.contexts,
            "prompt": result.prompt if print_prompt else None,
            "answer": result.answer,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if print_prompt:
        print(result.prompt)
        print()

    for i, c in enumerate(result.contexts, start=1):
        print(f"[{i}] {c}")
        print()


def _build_embedder(cfg: AppConfig):
    provider = cfg.embeddings.provider.lower().strip()
    if provider == "openai":
        return OpenAIEmbedder(model=cfg.embeddings.model, dimensions=cfg.embeddings.dimensions)
    if provider in {"sbert", "sentence_transformers"}:
        return SentenceTransformerEmbedder(model=cfg.embeddings.model)
    raise ValueError(f"不支持的 embeddings.provider: {cfg.embeddings.provider}")


def _build_store(cfg: AppConfig, *, reset: bool):
    provider = cfg.vector_store.provider.lower().strip()
    if provider == "chroma":
        persist_path = cfg.resolve_path(cfg.vector_store.persist_path)
        store = ChromaVectorStore(
            persist_path=persist_path,
            collection_name=cfg.vector_store.collection_name,
        )
        if reset:
            print("当前实现未提供 reset；如需重置请删除 db 目录后重建。", file=sys.stderr)
        return store
    raise ValueError(f"不支持的 vector_store.provider: {cfg.vector_store.provider}")


def _build_reranker(cfg: AppConfig):
    if not cfg.rerank.enabled:
        return None
    provider = cfg.rerank.provider.lower().strip()
    if provider in {"cross_encoder", "cross-encoder"}:
        return CrossEncoderReranker(model=cfg.rerank.model)
    raise ValueError(f"不支持的 rerank.provider: {cfg.rerank.provider}")

