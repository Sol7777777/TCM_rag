from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .config import AppConfig
from .errors import DependencyNotInstalledError
from .ingest.pdf import extract_paragraphs_from_pdf
from .modelscope import snapshot_download


@dataclass(frozen=True)
class LlamaIndexQueryResult:
    query: str
    answer: str
    contexts: list[str]
    formatted_prompt: Optional[str]


def download_models(
    cfg: AppConfig, *, download_embedding: bool = True, download_llm: bool = True
) -> dict[str, Optional[str]]:
    result: dict[str, Optional[str]] = {"embedding_dir": None, "llm_dir": None}
    if download_embedding:
        result["embedding_dir"] = snapshot_download(
            model_id=cfg.llamaindex.embedding_model_id, cache_dir=cfg.resolve_path(cfg.modelscope.cache_dir)
        )
    if download_llm:
        result["llm_dir"] = snapshot_download(
            model_id=cfg.llamaindex.llm_model_id, cache_dir=cfg.resolve_path(cfg.modelscope.cache_dir)
        )
    return result


def build_index(cfg: AppConfig, *, page_numbers: Optional[list[int]] = None) -> str:
    li = _li()
    documents_dir = cfg.resolve_path(cfg.llamaindex.documents_dir)
    required_exts = list(cfg.llamaindex.required_exts)
    persist_dir = cfg.resolve_path(cfg.llamaindex.persist_dir)

    model_dirs = download_models(cfg, download_embedding=True, download_llm=False)
    _configure_settings(
        li,
        embedding_model_dir=str(model_dirs["embedding_dir"]),
        llm_model_dir=None,
        cfg=cfg,
        enable_debug=False,
        enable_llm=False,
    )

    VectorStoreIndex = getattr(li, "VectorStoreIndex")
    SentenceSplitter = _import("llama_index.core.node_parser", extra_name="llamaindex").SentenceSplitter

    documents = _load_documents_for_index(
        cfg,
        li=li,
        documents_dir=documents_dir,
        required_exts=required_exts,
        page_numbers=page_numbers,
    )
    index = VectorStoreIndex.from_documents(
        documents, transformations=[SentenceSplitter(chunk_size=int(cfg.llamaindex.chunk_size))]
    )
    index.storage_context.persist(persist_dir=persist_dir)
    return persist_dir


def query(cfg: AppConfig, *, user_query: str, include_contexts: bool, enable_debug: bool) -> LlamaIndexQueryResult:
    query_engine, llama_debug = open_query_engine(cfg, enable_debug=bool(enable_debug))

    contexts: list[str] = []
    if include_contexts:
        contexts = retrieve_contexts(cfg, query_engine=query_engine, user_query=user_query)

    response = query_engine.query(user_query)
    answer = _clean_answer_text(str(response))
    formatted_prompt: Optional[str] = None
    if llama_debug is not None:
        event_pairs = llama_debug.get_llm_inputs_outputs()
        if event_pairs:
            payload = getattr(event_pairs[0][1], "payload", {}) or {}
            formatted_prompt = payload.get("formatted_prompt")

    return LlamaIndexQueryResult(
        query=user_query,
        answer=answer,
        contexts=contexts,
        formatted_prompt=formatted_prompt,
    )


def _clean_answer_text(text: str) -> str:
    s = str(text).strip()
    if s.endswith("[/INST]"):
        s = s[: -len("[/INST]")].rstrip()
    if s.endswith("</s>"):
        s = s[: -len("</s>")].rstrip()
    return s


def open_query_engine(cfg: AppConfig, *, enable_debug: bool):
    li = _li()
    storage_context_mod = _import("llama_index.core", extra_name="llamaindex")
    StorageContext = getattr(storage_context_mod, "StorageContext")
    load_index_from_storage = getattr(storage_context_mod, "load_index_from_storage")

    model_dirs = download_models(cfg, download_embedding=True, download_llm=True)
    llama_debug = _configure_settings(
        li,
        embedding_model_dir=str(model_dirs["embedding_dir"]),
        llm_model_dir=str(model_dirs["llm_dir"]),
        cfg=cfg,
        enable_debug=bool(enable_debug),
        enable_llm=True,
    )

    persist_dir = cfg.resolve_path(cfg.llamaindex.persist_dir)
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine(similarity_top_k=int(cfg.llamaindex.similarity_top_k))

    PromptTemplate = getattr(li, "PromptTemplate")
    qa_prompt = PromptTemplate(cfg.llamaindex.qa_prompt)
    refine_prompt = PromptTemplate(cfg.llamaindex.refine_prompt)
    query_engine.update_prompts(
        {
            "response_synthesizer:text_qa_template": qa_prompt,
            "response_synthesizer:refine_template": refine_prompt,
        }
    )
    return query_engine, llama_debug


def retrieve_contexts(cfg: AppConfig, *, query_engine: Any, user_query: str) -> list[str]:
    li = _li()
    QueryBundle = getattr(li, "QueryBundle")
    MetadataMode = _import("llama_index.core.schema", extra_name="llamaindex").MetadataMode

    nodes = query_engine.retrieve(QueryBundle(user_query))
    contexts: list[str] = []
    for n in nodes:
        node = getattr(n, "node", None)
        if node is None:
            continue
        get_content = getattr(node, "get_content", None)
        if get_content is None:
            continue
        contexts.append(str(get_content(metadata_mode=MetadataMode.LLM)))
    return contexts


def open_retriever(cfg: AppConfig):
    li = _li()
    storage_context_mod = _import("llama_index.core", extra_name="llamaindex")
    StorageContext = getattr(storage_context_mod, "StorageContext")
    load_index_from_storage = getattr(storage_context_mod, "load_index_from_storage")

    model_dirs = download_models(cfg, download_embedding=True, download_llm=False)
    _configure_settings(
        li,
        embedding_model_dir=str(model_dirs["embedding_dir"]),
        llm_model_dir=None,
        cfg=cfg,
        enable_debug=False,
        enable_llm=False,
    )

    persist_dir = cfg.resolve_path(cfg.llamaindex.persist_dir)
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
    retriever = index.as_retriever(similarity_top_k=int(cfg.llamaindex.similarity_top_k))
    return retriever


def retrieve_contexts_from_retriever(cfg: AppConfig, *, retriever: Any, user_query: str) -> list[str]:
    li = _li()
    QueryBundle = getattr(li, "QueryBundle")
    MetadataMode = _import("llama_index.core.schema", extra_name="llamaindex").MetadataMode

    nodes = retriever.retrieve(QueryBundle(user_query))
    contexts: list[str] = []
    for n in nodes:
        node = getattr(n, "node", None)
        if node is None:
            continue
        get_content = getattr(node, "get_content", None)
        if get_content is None:
            continue
        contexts.append(str(get_content(metadata_mode=MetadataMode.LLM)))
    return contexts


def _configure_settings(
    li: Any,
    *,
    embedding_model_dir: str,
    llm_model_dir: Optional[str],
    cfg: AppConfig,
    enable_debug: bool,
    enable_llm: bool,
):
    torch = _optional_import("torch", extra_name="torch")

    Settings = getattr(li, "Settings")
    PromptTemplate = getattr(li, "PromptTemplate")

    SYSTEM_PROMPT = cfg.llamaindex.system_prompt
    query_wrapper_prompt = PromptTemplate(
        "[INST]<<SYS>>\n" + SYSTEM_PROMPT + "<</SYS>>\n\n{query_str}[/INST] "
    )

    if enable_llm:
        if not llm_model_dir:
            raise ValueError("llm_model_dir 不能为空")
        HuggingFaceLLM = _import("llama_index.llms.huggingface", extra_name="llamaindex").HuggingFaceLLM
        llm = HuggingFaceLLM(
            context_window=int(cfg.llamaindex.context_window),
            max_new_tokens=int(cfg.llamaindex.max_new_tokens),
            generate_kwargs={
                "temperature": float(cfg.llamaindex.temperature),
                "do_sample": bool(cfg.llamaindex.do_sample),
            },
            query_wrapper_prompt=query_wrapper_prompt,
            tokenizer_name=str(llm_model_dir),
            model_name=str(llm_model_dir),
            device_map=str(cfg.llamaindex.device_map),
            model_kwargs={"torch_dtype": _torch_dtype(torch, cfg.llamaindex.torch_dtype)},
        )
        Settings.llm = llm

    HuggingFaceEmbedding = _import(
        "llama_index.embeddings.huggingface", extra_name="llamaindex"
    ).HuggingFaceEmbedding
    Settings.embed_model = HuggingFaceEmbedding(model_name=str(embedding_model_dir))

    llama_debug = None
    if enable_debug:
        callbacks = _import("llama_index.core.callbacks", extra_name="llamaindex")
        LlamaDebugHandler = getattr(callbacks, "LlamaDebugHandler")
        CallbackManager = getattr(callbacks, "CallbackManager")
        llama_debug = LlamaDebugHandler(print_trace_on_end=True)
        Settings.callback_manager = CallbackManager([llama_debug])

    return llama_debug


def _load_documents_for_index(
    cfg: AppConfig,
    *,
    li: Any,
    documents_dir: str,
    required_exts: list[str],
    page_numbers: Optional[list[int]],
):
    SimpleDirectoryReader = getattr(li, "SimpleDirectoryReader")
    Document = getattr(_import("llama_index.core", extra_name="llamaindex"), "Document")

    doc_dir = Path(documents_dir)
    if doc_dir.exists() and doc_dir.is_dir():
        docs = SimpleDirectoryReader(str(doc_dir), required_exts=required_exts).load_data()
        if docs:
            return docs

    pdf_path = cfg.resolve_path(cfg.ingest.pdf_path)
    paragraphs = extract_paragraphs_from_pdf(
        pdf_path,
        page_numbers=page_numbers if page_numbers is not None else cfg.ingest.page_numbers,
        min_line_length=cfg.ingest.min_line_length,
    )
    text = "\n\n".join(paragraphs)
    return [Document(text=text, metadata={"source": cfg.ingest.pdf_path})]


def _li():
    return _import("llama_index.core", extra_name="llamaindex")


def _torch_dtype(torch: Any, dtype: str):
    key = str(dtype).strip().lower()
    if key in {"float16", "fp16"}:
        return torch.float16
    if key in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if key in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"不支持的 torch_dtype: {dtype}")


def _import(module_name: str, *, extra_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        raise DependencyNotInstalledError(
            f"缺少可选依赖: {module_name}，请安装 extra: {extra_name}"
        ) from e


def _optional_import(module_name: str, *, extra_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        raise DependencyNotInstalledError(
            f"缺少可选依赖: {module_name}，请安装 extra: {extra_name}"
        ) from e
