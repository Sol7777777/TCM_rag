from __future__ import annotations

import re
from typing import Iterable


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？；?!])")


def split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p and p.strip()]


def split_text(
    paragraphs: Iterable[str],
    *,
    chunk_size: int = 350,
    overlap_size: int = 120,
) -> list[str]:
    sentences: list[str] = []
    for p in paragraphs:
        sentences.extend(split_sentences(p))

    chunks: list[str] = []
    i = 0
    while i < len(sentences):
        chunk = sentences[i]
        overlap = ""

        prev = i - 1
        while prev >= 0 and len(sentences[prev]) + len(overlap) <= overlap_size:
            overlap = f"{sentences[prev]} {overlap}".strip()
            prev -= 1

        if overlap:
            chunk = f"{overlap} {chunk}".strip()

        nxt = i + 1
        while nxt < len(sentences) and len(sentences[nxt]) + len(chunk) <= chunk_size:
            chunk = f"{chunk} {sentences[nxt]}".strip()
            nxt += 1

        chunks.append(chunk)
        i = nxt

    return chunks

