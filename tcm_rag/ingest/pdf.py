from __future__ import annotations

import importlib
import io
from typing import Iterable, Optional

from ..errors import DependencyNotInstalledError


def extract_paragraphs_from_pdf(
    filename: str,
    *,
    page_numbers: Optional[Iterable[int]] = None,
    min_line_length: int = 10,
) -> list[str]:
    paragraphs: list[str] = []
    buffer = ""
    full_text = ""

    try:
        full_text = _extract_text_with_pdfminer(filename, page_numbers=page_numbers)
    except DependencyNotInstalledError:
        full_text = ""

    if not full_text.strip() or _looks_garbled_text(full_text):
        full_text = _extract_text_with_fitz(filename, page_numbers=page_numbers)

    if not full_text.strip() or _looks_garbled_text(full_text):
        full_text = _extract_text_with_fitz_ocr(filename, page_numbers=page_numbers)

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


def _looks_garbled_text(text: str) -> bool:
    sample = "".join(ch for ch in str(text)[:5000] if not ch.isspace())
    if len(sample) < 200:
        return True
    cjk = 0
    alnum = 0
    punct = 0
    punct_chars = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
    for ch in sample:
        if "\u4e00" <= ch <= "\u9fff":
            cjk += 1
            continue
        if ch.isalnum():
            alnum += 1
            continue
        if ch in punct_chars:
            punct += 1
    meaningful = cjk + alnum
    if cjk == 0 and punct / len(sample) > 0.25:
        return True
    return meaningful / len(sample) < 0.20


def _extract_text_with_pdfminer(filename: str, *, page_numbers: Optional[Iterable[int]]):
    pdfminer_high_level = _optional_import("pdfminer.high_level", "pdf")
    pdfminer_layout = _optional_import("pdfminer.layout", "pdf")
    extract_pages = getattr(pdfminer_high_level, "extract_pages")
    LTTextContainer = getattr(pdfminer_layout, "LTTextContainer")

    page_set = set(page_numbers) if page_numbers is not None else None
    parts: list[str] = []
    for i, page_layout in enumerate(extract_pages(filename)):
        if page_set is not None and i not in page_set:
            continue
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                parts.append(element.get_text())
    return "\n".join(parts)


def _extract_text_with_fitz(filename: str, *, page_numbers: Optional[Iterable[int]]):
    fitz = _optional_import("fitz", "pdf")
    doc = fitz.open(filename)

    page_set = set(page_numbers) if page_numbers is not None else None
    parts: list[str] = []
    for i in range(doc.page_count):
        if page_set is not None and i not in page_set:
            continue
        page = doc.load_page(i)
        parts.append(str(page.get_text("text")))
    return "\n".join(parts)


def _extract_text_with_fitz_ocr(filename: str, *, page_numbers: Optional[Iterable[int]]):
    fitz = _optional_import("fitz", "pdf")
    Image = _optional_import("PIL.Image", "pdf")
    pytesseract = _optional_import("pytesseract", "pdf")

    doc = fitz.open(filename)
    page_set = set(page_numbers) if page_numbers is not None else None
    parts: list[str] = []
    for i in range(doc.page_count):
        if page_set is not None and i not in page_set:
            continue
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        parts.append(str(pytesseract.image_to_string(img, lang="chi_sim")))
    return "\n".join(parts)


def _optional_import(module_name: str, extra_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        raise DependencyNotInstalledError(
            f"缺少可选依赖: {module_name}，请安装 extra: {extra_name}"
        ) from e
