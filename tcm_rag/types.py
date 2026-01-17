from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


Embedding = Sequence[float]


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    text: str
    metadata: Mapping[str, Any]

