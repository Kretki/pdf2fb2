"""Roles of PP-DocLayoutV3 segment labels and per-label OCR settings.

Unknown labels are treated as body text.
"""

SKIPPED_LABELS = frozenset({"header", "footer", "header_image", "footer_image"})  # running heads, dropped
PAGE_NUMBER_LABELS = frozenset({"number"})  # printed page number
DROPPED_LABELS = frozenset({"inline_formula", "formula_number"})  # covered by parent text/formula
PICTURE_LABELS = frozenset({"image", "chart", "seal", "display_formula"})  # embedded as pictures
CAPTION_LABELS = frozenset({"figure_title", "table_title", "chart_title"})
FOOTNOTE_LABELS = frozenset({"footnote", "vision_footnote"})
DOCUMENT_TITLE_LABEL = "doc_title"
HEADING_LABELS = frozenset({"paragraph_title"})
TABLE_LABEL = "table"
ABSTRACT_LABEL = "abstract"

LABEL_COLORS = {
    "doc_title": (220, 20, 60), "paragraph_title": (255, 140, 0), "text": (30, 144, 255),
    "abstract": (0, 191, 255), "table": (34, 139, 34), "image": (148, 0, 211),
    "chart": (199, 21, 133), "figure_title": (218, 165, 32), "table_title": (218, 165, 32),
    "chart_title": (218, 165, 32), "display_formula": (0, 128, 128), "header": (128, 128, 0),
    "footer": (128, 128, 0), "number": (112, 128, 144), "footnote": (160, 82, 45),
    "seal": (178, 34, 34), "reference": (72, 61, 139), "aside_text": (100, 149, 237),
}
DEFAULT_LABEL_COLOR = (128, 128, 128)

QIANFAN_PROMPT = "Parse this document to Markdown."
OCR_TOKEN_LIMITS = {"table": 8192}
DEFAULT_OCR_TOKEN_LIMIT = 2048


def requires_ocr(segment_label: str) -> bool:
    """Tell whether a segment must be read by the OCR model.

    Usage:
        Call when collecting OCR work and when applying OCR results.

    Args:
        segment_label: PP-DocLayoutV3 label of the segment.

    Returns:
        False for running heads and picture-like segments, True otherwise.
    """
    return segment_label not in SKIPPED_LABELS and segment_label not in PICTURE_LABELS


def ocr_token_limit(segment_label: str) -> int:
    """Return the generation budget for a segment label.

    Args:
        segment_label: PP-DocLayoutV3 label of the segment.

    Returns:
        Maximum number of new tokens (8192 for tables, 2048 otherwise).
    """
    return OCR_TOKEN_LIMITS.get(segment_label, DEFAULT_OCR_TOKEN_LIMIT)
