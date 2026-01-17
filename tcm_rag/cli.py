from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import AppConfig, load_config
from .embeddings.openai_embedder import OpenAIEmbedder
from .embeddings.sbert import SentenceTransformerEmbedder
from .ingest.pdf import extract_paragraphs_from_pdf
from .llamaindex_rag import build_index as li_build_index
from .llamaindex_rag import download_models as li_download_models
from .llamaindex_rag import open_query_engine as li_open_query_engine
from .llamaindex_rag import query as li_query
from .llamaindex_rag import retrieve_contexts as li_retrieve_contexts
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

    p_download = sub.add_parser("download-models", help="使用ModelScope下载本地模型权重")

    p_li_build = sub.add_parser("li-build", help="用LlamaIndex构建并持久化索引")

    p_li_query = sub.add_parser("li-query", help="用LlamaIndex检索问答")
    p_li_query.add_argument("text", help="用户问题")
    p_li_query.add_argument("--with-contexts", action="store_true", help="输出被检索的文档片段")
    p_li_query.add_argument("--debug", action="store_true", help="输出 formatted_prompt")
    p_li_query.add_argument("--json", action="store_true")

    p_li_retrieve = sub.add_parser("li-retrieve", help="用LlamaIndex仅检索不生成")
    p_li_retrieve.add_argument("text", help="用户问题")
    p_li_retrieve.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.command == "ingest":
        _cmd_ingest(cfg, reset=bool(args.reset))
        return 0
    if args.command == "query":
        _cmd_query(cfg, query=args.text, top_k=int(args.top_k), print_prompt=bool(args.print_prompt), as_json=bool(args.json))
        return 0
    if args.command == "download-models":
        _cmd_download_models(cfg)
        return 0
    if args.command == "li-build":
        _cmd_li_build(cfg)
        return 0
    if args.command == "li-query":
        _cmd_li_query(
            cfg,
            query=args.text,
            include_contexts=bool(args.with_contexts),
            enable_debug=bool(args.debug),
            as_json=bool(args.json),
        )
        return 0
    if args.command == "li-retrieve":
        _cmd_li_retrieve(cfg, query=args.text, as_json=bool(args.json))
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


def _cmd_download_models(cfg: AppConfig) -> None:
    paths = li_download_models(cfg)
    print(json.dumps(paths, ensure_ascii=False, indent=2))


def _cmd_li_build(cfg: AppConfig) -> None:
    persist_dir = li_build_index(cfg)
    print(f"已写入并持久化索引: {persist_dir}")


def _cmd_li_query(
    cfg: AppConfig,
    *,
    query: str,
    include_contexts: bool,
    enable_debug: bool,
    as_json: bool,
) -> None:
    result = li_query(cfg, user_query=query, include_contexts=include_contexts, enable_debug=enable_debug)

    if as_json:
        payload = {
            "query": result.query,
            "contexts": result.contexts,
            "answer": result.answer,
            "formatted_prompt": result.formatted_prompt,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if result.formatted_prompt:
        print(result.formatted_prompt)
        print()

    if result.contexts:
        print("-" * 10 + "ref" + "-" * 10)
        for i, c in enumerate(result.contexts):
            print("*" * 10 + f"chunk {i} start" + "*" * 10)
            print(c)
            print("*" * 10 + f"chunk {i} end" + "*" * 10)
        print("-" * 10 + "ref" + "-" * 10)
        print()

    print(result.answer)


def _cmd_li_retrieve(cfg: AppConfig, *, query: str, as_json: bool) -> None:
    query_engine, _ = li_open_query_engine(cfg, enable_debug=False)
    contexts = li_retrieve_contexts(cfg, query_engine=query_engine, user_query=query)
    if as_json:
        print(json.dumps({"query": query, "contexts": contexts}, ensure_ascii=False, indent=2))
        return
    print("-" * 10 + "ref" + "-" * 10)
    for i, c in enumerate(contexts):
        print("*" * 10 + f"chunk {i} start" + "*" * 10)
        print(c)
        print("*" * 10 + f"chunk {i} end" + "*" * 10)
    print("-" * 10 + "ref" + "-" * 10)


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
