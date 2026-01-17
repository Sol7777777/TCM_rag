from __future__ import annotations

import importlib
from pathlib import Path

from .errors import DependencyNotInstalledError


def snapshot_download(*, model_id: str, cache_dir: str) -> str:
    ms = _optional_import("modelscope", extra_name="modelscope")
    fn = getattr(ms, "snapshot_download", None)
    if fn is None:
        raise DependencyNotInstalledError("modelscope.snapshot_download 不存在")
    local_dir = fn(model_id=str(model_id), cache_dir=str(cache_dir))
    return str(Path(local_dir).resolve())


def _optional_import(module_name: str, *, extra_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        raise DependencyNotInstalledError(
            f"缺少可选依赖: {module_name}，请安装 extra: {extra_name}"
        ) from e

