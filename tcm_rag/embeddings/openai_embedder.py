from __future__ import annotations

import importlib
from typing import Optional, Sequence

from ..config import get_env
from ..errors import DependencyNotInstalledError


class OpenAIEmbedder:
    def __init__(self, *, model: str, dimensions: Optional[int] = None):
        self._model = model
        self._dimensions = dimensions
        self._client = _create_openai_client()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if self._dimensions is None:
            data = self._client.embeddings.create(input=list(texts), model=self._model).data
        else:
            data = self._client.embeddings.create(
                input=list(texts), model=self._model, dimensions=int(self._dimensions)
            ).data
        return [x.embedding for x in data]


def _create_openai_client():
    try:
        openai = importlib.import_module("openai")
    except Exception as e:
        raise DependencyNotInstalledError("缺少可选依赖: openai，请安装 extra: openai") from e

    api_key = get_env("OPENAI_API_KEY")
    if not api_key:
        raise DependencyNotInstalledError("缺少环境变量 OPENAI_API_KEY")

    base_url = get_env("OPENAI_BASE_URL")
    if base_url:
        return openai.OpenAI(api_key=api_key, base_url=base_url)
    return openai.OpenAI(api_key=api_key)

