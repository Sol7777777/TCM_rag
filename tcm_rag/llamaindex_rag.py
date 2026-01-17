from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Optional

from .config import AppConfig
from .errors import DependencyNotInstalledError
from .modelscope import snapshot_download


@dataclass(frozen=True)
class LlamaIndexQueryResult:
    query: str
    answer: str
    contexts: list[str]
    formatted_prompt: Optional[str]


def download_models(cfg: AppConfig) -> dict[str, str]:
    embedding_dir = snapshot_download(
        model_id=cfg.llamaindex.embedding_model_id, cache_dir=cfg.resolve_path(cfg.modelscope.cache_dir)
    )
    llm_dir = snapshot_download(
        model_id=cfg.llamaindex.llm_model_id, cache_dir=cfg.resolve_path(cfg.modelscope.cache_dir)
    )
    return {"embedding_dir": embedding_dir, "llm_dir": llm_dir}


def build_index(cfg: AppConfig) -> str:
    li = _li()
    documents_dir = cfg.resolve_path(cfg.llamaindex.documents_dir)
    required_exts = list(cfg.llamaindex.required_exts)
    persist_dir = cfg.resolve_path(cfg.llamaindex.persist_dir)

    model_dirs = download_models(cfg)
    _configure_settings(
        li,
        embedding_model_dir=model_dirs["embedding_dir"],
        llm_model_dir=model_dirs["llm_dir"],
        cfg=cfg,
        enable_debug=False,
    )

    SimpleDirectoryReader = getattr(li, "SimpleDirectoryReader")
    VectorStoreIndex = getattr(li, "VectorStoreIndex")
    SentenceSplitter = _import("llama_index.core.node_parser", extra_name="llamaindex").SentenceSplitter

    documents = SimpleDirectoryReader(documents_dir, required_exts=required_exts).load_data()
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
    formatted_prompt: Optional[str] = None
    if llama_debug is not None:
        event_pairs = llama_debug.get_llm_inputs_outputs()
        if event_pairs:
            payload = getattr(event_pairs[0][1], "payload", {}) or {}
            formatted_prompt = payload.get("formatted_prompt")

    return LlamaIndexQueryResult(
        query=user_query,
        answer=str(response),
        contexts=contexts,
        formatted_prompt=formatted_prompt,
    )


def open_query_engine(cfg: AppConfig, *, enable_debug: bool):
    li = _li()
    storage_context_mod = _import("llama_index.core", extra_name="llamaindex")
    StorageContext = getattr(storage_context_mod, "StorageContext")
    load_index_from_storage = getattr(storage_context_mod, "load_index_from_storage")

    model_dirs = download_models(cfg)
    llama_debug = _configure_settings(
        li,
        embedding_model_dir=model_dirs["embedding_dir"],
        llm_model_dir=model_dirs["llm_dir"],
        cfg=cfg,
        enable_debug=bool(enable_debug),
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


def _configure_settings(
    li: Any,
    *,
    embedding_model_dir: str,
    llm_model_dir: str,
    cfg: AppConfig,
    enable_debug: bool,
):
    torch = _optional_import("torch", extra_name="torch")

    Settings = getattr(li, "Settings")
    PromptTemplate = getattr(li, "PromptTemplate")

    SYSTEM_PROMPT = cfg.llamaindex.system_prompt
    query_wrapper_prompt = PromptTemplate(
        "[INST]<<SYS>>\n" + SYSTEM_PROMPT + "<</SYS>>\n\n{query_str}[/INST] "
    )

    HuggingFaceLLM = _import("llama_index.llms.huggingface", extra_name="llamaindex").HuggingFaceLLM
    llm = HuggingFaceLLM(
        context_window=int(cfg.llamaindex.context_window),
        max_new_tokens=int(cfg.llamaindex.max_new_tokens),
        generate_kwargs={"temperature": float(cfg.llamaindex.temperature), "do_sample": bool(cfg.llamaindex.do_sample)},
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
