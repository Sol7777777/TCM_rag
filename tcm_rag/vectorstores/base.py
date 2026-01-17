from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence


class VectorStore(Protocol):
    def add(
        self,
        *,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[Mapping[str, Any]],
    ) -> None:
        raise NotImplementedError

    def query(self, *, query_embedding: Sequence[float], n_results: int) -> dict[str, Any]:
        raise NotImplementedError

