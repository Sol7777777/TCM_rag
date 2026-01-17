from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..embeddings.base import Embedder
from ..prompting.templates import build_qa_prompt
from ..rerank.base import Reranker
from ..vectorstores.base import VectorStore


@dataclass(frozen=True)
class RAGResponse:
    query: str
    contexts: list[str]
    prompt: str
    answer: Optional[str]
    raw_search: dict[str, Any]


class RAGPipeline:
    def __init__(
        self,
        *,
        vector_store: VectorStore,
        embedder: Embedder,
        llm: Optional[Callable[[str], str]] = None,
        reranker: Optional[Reranker] = None,
    ):
        self._vector_store = vector_store
        self._embedder = embedder
        self._llm = llm
        self._reranker = reranker

    def query(self, *, user_query: str, top_k: int = 5) -> RAGResponse:
        q_emb = self._embedder.embed([user_query])[0]
        raw = self._vector_store.query(query_embedding=q_emb, n_results=int(top_k))
        contexts = (raw.get("documents") or [[]])[0]

        if self._reranker and contexts:
            ranked = self._reranker.rerank(query=user_query, documents=contexts)
            contexts = [d for _, d in ranked]

        context_text = "\n\n".join(contexts)
        prompt = build_qa_prompt(context=context_text, query=user_query)

        answer = self._llm(prompt) if self._llm else None
        return RAGResponse(
            query=user_query,
            contexts=list(contexts),
            prompt=prompt,
            answer=answer,
            raw_search=raw,
        )

