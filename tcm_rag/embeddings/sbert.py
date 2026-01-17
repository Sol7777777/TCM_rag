from __future__ import annotations

import importlib
from typing import Sequence

from ..errors import DependencyNotInstalledError


class SentenceTransformerEmbedder:
    def __init__(self, *, model: str):
        self._model = model
        self._st = _load_st_model(model)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._st.encode(list(texts), normalize_embeddings=True)
        return [v.tolist() for v in vectors]


def _load_st_model(model: str):
    try:
        st = importlib.import_module("sentence_transformers")
    except Exception as e:
        raise DependencyNotInstalledError(
            "缺少可选依赖: sentence-transformers，请安装 extra: sbert"
        ) from e
    SentenceTransformer = getattr(st, "SentenceTransformer")
    return SentenceTransformer(model)

