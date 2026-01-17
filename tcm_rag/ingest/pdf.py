from __future__ import annotations

import importlib
from typing import Iterable, Optional

from ..errors import DependencyNotInstalledError


def extract_paragraphs_from_pdf(
    filename: str,
    *,
    page_numbers: Optional[Iterable[int]] = None,
    min_line_length: int = 10,
) -> list[str]:
    pdfminer_high_level = _optional_import("pdfminer.high_level", "pdf")
    pdfminer_layout = _optional_import("pdfminer.layout", "pdf")
    extract_pages = getattr(pdfminer_high_level, "extract_pages")
    LTTextContainer = getattr(pdfminer_layout, "LTTextContainer")

    paragraphs: list[str] = []
    buffer = ""
    full_text = ""

    page_set = set(page_numbers) if page_numbers is not None else None
    for i, page_layout in enumerate(extract_pages(filename)):
        if page_set is not None and i not in page_set:
            continue
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                full_text += element.get_text() + "\n"

    lines = full_text.split("\n")
    for text in lines:
        if len(text) >= min_line_length:
            buffer += (" " + text) if not text.endswith("-") else text.strip("-")
        elif buffer:
            paragraphs.append(buffer.strip())
            buffer = ""

    if buffer:
        paragraphs.append(buffer.strip())

    return paragraphs


def _optional_import(module_name: str, extra_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        raise DependencyNotInstalledError(
            f"缺少可选依赖: {module_name}，请安装 extra: {extra_name}"
        ) from e

