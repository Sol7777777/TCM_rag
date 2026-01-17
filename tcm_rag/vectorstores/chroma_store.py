from __future__ import annotations

import importlib
from typing import Any, Mapping, Sequence

from ..errors import DependencyNotInstalledError


class ChromaVectorStore:
    def __init__(self, *, persist_path: str, collection_name: str):
        chromadb = _optional_import("chromadb", "chroma")
        settings_mod = _optional_import("chromadb.config", "chroma")
        Settings = getattr(settings_mod, "Settings")

        self._client = chromadb.PersistentClient(
            path=persist_path, settings=Settings(allow_reset=False)
        )
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def add(
        self,
        *,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[Mapping[str, Any]],
    ) -> None:
        self._collection.add(
            ids=list(ids),
            embeddings=[list(x) for x in embeddings],
            documents=list(documents),
            metadatas=[dict(m) for m in metadatas],
        )

    def query(self, *, query_embedding: Sequence[float], n_results: int) -> dict[str, Any]:
        return self._collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=int(n_results),
        )


def _optional_import(module_name: str, extra_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        raise DependencyNotInstalledError(
            f"缺少可选依赖: {module_name}，请安装 extra: {extra_name}"
        ) from e

