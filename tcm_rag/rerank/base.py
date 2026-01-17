from __future__ import annotations

from typing import Protocol, Sequence


class Reranker(Protocol):
    def rerank(self, *, query: str, documents: Sequence[str]) -> list[tuple[float, str]]:
        raise NotImplementedError

