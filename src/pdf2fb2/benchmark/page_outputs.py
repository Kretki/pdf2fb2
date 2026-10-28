"""Text of every converted page, at two levels of the pipeline.

* ``ocr`` level: ``segments.json``, the text segments of a page in reading order after OCR, the verifier
  feedback and word styles, before proofreading. Running heads, page numbers, handwriting and pictures are
  left out, as in the FB2.
* ``final`` level: the FB2 main body split at the invisible page anchors (``id="page_N"``), after
  proofreading. A paragraph running across a page break counts for the page it starts on.
"""

import html
import json
import re
from pathlib import Path

from pdf2fb2.segments.label_roles import is_book_text

SEGMENTS_FILE = "segments.json"
MAX_OUTPUT_BYTES = 500_000_000
STYLE_TAG_PATTERN = re.compile(r"</?(?:b|i|s)>")
BODY_PATTERN = re.compile(r"<body>(.*?)</body>", re.DOTALL)
PAGE_ANCHOR_PATTERN = re.compile(r'<[^<>]{0,200}\bid="page_(\d{1,6})"[^<>]{0,200}>')
XML_TAG_PATTERN = re.compile(r"<[^<>]{0,500}>")


def segment_text(page_segment: dict) -> str:
    """Unstyled text of a segment.

    Args:
        page_segment: Segment record.

    Returns:
        ``ocr_text`` when styles were applied, otherwise ``text`` with style tags removed.
    """
    return page_segment.get("ocr_text") or STYLE_TAG_PATTERN.sub("", page_segment.get("text") or "")


def ocr_page_texts(conversion_directory: Path) -> dict[int, str]:
    """Page number -> text of the book's segments.

    Args:
        conversion_directory: Output folder of one conversion.

    Returns:
        Texts keyed by the 1-based page of the converted PDF (empty when ``segments.json`` is missing).
    """
    segments_path = conversion_directory / SEGMENTS_FILE
    if not segments_path.exists() or segments_path.stat().st_size > MAX_OUTPUT_BYTES:
        return {}
    page_records = json.loads(segments_path.read_text(encoding="utf-8"))
    return {page_record["page"]: "\n".join(segment_text(page_segment) for page_segment in page_record["segments"]
                                           if is_book_text(page_segment))
            for page_record in page_records}


def final_page_texts(fb2_path: Path) -> dict[int, str]:
    """Page number -> FB2 text from its anchor to the next one.

    Args:
        fb2_path: FB2 written by the conversion (anchor page markers).

    Returns:
        Plain texts keyed by the 1-based page of the converted PDF (empty when the file is missing).
    """
    if not fb2_path.exists() or fb2_path.stat().st_size > MAX_OUTPUT_BYTES:
        return {}
    body_match = BODY_PATTERN.search(fb2_path.read_text(encoding="utf-8"))
    if body_match is None:
        return {}
    body_pieces = PAGE_ANCHOR_PATTERN.split(body_match.group(1))
    return {int(body_pieces[piece_index]): html.unescape(XML_TAG_PATTERN.sub(" ", body_pieces[piece_index + 1]))
            for piece_index in range(1, len(body_pieces) - 1, 2)}
