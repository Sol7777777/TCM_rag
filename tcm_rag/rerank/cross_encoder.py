from __future__ import annotations

import importlib
from typing import Sequence

from ..errors import DependencyNotInstalledError


class CrossEncoderReranker:
    def __init__(self, *, model: str, max_length: int = 512):
        self._model = _load_ce_model(model, max_length=max_length)

    def rerank(self, *, query: str, documents: Sequence[str]) -> list[tuple[float, str]]:
        pairs = [(query, d) for d in documents]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(scores, documents), key=lambda x: float(x[0]), reverse=True)
        return [(float(s), d) for s, d in ranked]


def _load_ce_model(model: str, *, max_length: int):
    try:
        st = importlib.import_module("sentence_transformers")
    except Exception as e:
        raise DependencyNotInstalledError(
            "缺少可选依赖: sentence-transformers，请安装 extra: sbert"
        ) from e
    CrossEncoder = getattr(st, "CrossEncoder")
    return CrossEncoder(model, max_length=int(max_length))

