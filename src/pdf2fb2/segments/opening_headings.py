"""Opening-heading check after OCR: a short centred line set apart above the first paragraph is a heading.

Chapter and preface openings print their title alone on one line, centred, with a wider gap to the text below
than the gap between paragraphs. When the layout model labels such a line plain ``text`` (or when the
residual-text rescan adds it as ``text``) it becomes ``paragraph_title`` if:

* it is the first text block of its page (running heads and page numbers do not count);
* it is short: not longer than the 10th percentile of the book's paragraphs (see ``heading_statistics``);
* it is a single line: its box is at most ``SINGLE_LINE_FACTOR`` usual glyph heights tall;
* it is centred on the body-text column of its page group;
* the next text block starts at least ``OPENING_GAP_FACTOR`` usual glyph heights below it.

Unlike ``heading_statistics.mark_sparse_headings`` this does not need a near-empty page.
"""

from pdf2fb2.segments.heading_statistics import (
    CENTER_TOLERANCE_RATIO,
    FALLBACK_GLYPH_HEIGHT,
    HEADING_LABEL,
    BookTextStatistics,
    collect_text_statistics,
    is_text_block,
    visible_length,
)
from pdf2fb2.segments.label_roles import BODY_TEXT_LABEL
from pdf2fb2.segments.position_statistics import center_ratio

SINGLE_LINE_FACTOR = 3.0  # box height of one line, in usual glyph heights
OPENING_GAP_FACTOR = 3.0  # gap to the paragraph below, in usual glyph heights


def usual_glyph_height(text_statistics: BookTextStatistics) -> float:
    """Usual glyph height of the book.

    Args:
        text_statistics: Statistics of the book.

    Returns:
        Typical glyph height in pixels, or ``FALLBACK_GLYPH_HEIGHT`` when not measured.
    """
    font_statistics = text_statistics.font_statistics
    return font_statistics.typical_height if font_statistics is not None else FALLBACK_GLYPH_HEIGHT


def gap_below(page_segment: dict, page_record: dict) -> float | None:
    """Gap between a segment and the text below it, when nothing is above it.

    Args:
        page_segment: Candidate segment.
        page_record: Its page record.

    Returns:
        Distance in pixels to the nearest text block starting below the segment; None when another text block
        starts above it or no text follows.
    """
    segment_box = page_segment["bbox"]
    other_boxes = [other_segment["bbox"] for other_segment in page_record["segments"]
                   if other_segment is not page_segment and is_text_block(other_segment)]
    if any(other_box[1] < segment_box[1] for other_box in other_boxes):
        return None
    following_tops = [other_box[1] for other_box in other_boxes if other_box[1] >= segment_box[3]]
    return min(following_tops) - segment_box[3] if following_tops else None


def is_opening_heading(page_segment: dict, page_record: dict, text_statistics: BookTextStatistics) -> bool:
    """Tell whether a text segment is the title line of an opening page.

    Args:
        page_segment: Segment of the page.
        page_record: Its page record.
        text_statistics: Statistics of the book.

    Returns:
        True for short, single-line, centred ``text`` set apart above the first paragraph of the page.
    """
    page_width = float(page_record["size"][0])
    column_center = text_statistics.column_centers.get(page_record["page"] % 2)
    if page_segment["label"] != BODY_TEXT_LABEL or column_center is None or page_width <= 0:
        return False
    glyph_height = usual_glyph_height(text_statistics)
    segment_box = page_segment["bbox"]
    following_gap = gap_below(page_segment, page_record)
    return (following_gap is not None and following_gap >= OPENING_GAP_FACTOR * glyph_height
            and segment_box[3] - segment_box[1] <= SINGLE_LINE_FACTOR * glyph_height
            and 0 < visible_length(page_segment) <= text_statistics.short_text_length
            and abs(center_ratio(segment_box, page_width) - column_center) <= CENTER_TOLERANCE_RATIO)


def mark_opening_headings(page_records: list[dict]) -> int:
    """Relabel the title lines of opening pages as headings.

    Usage:
        Call on the OCR'd page records, before word styles are applied and the book flow is built.

    Args:
        page_records: OCR'd page records (mutated: labels rewritten, ``original_label`` kept).

    Returns:
        Number of relabelled segments.
    """
    text_statistics = collect_text_statistics(page_records)
    if text_statistics is None:
        return 0
    heading_segments = [page_segment for page_record in page_records for page_segment in page_record["segments"]
                        if is_opening_heading(page_segment, page_record, text_statistics)]
    for heading_segment in heading_segments:
        heading_segment["original_label"] = heading_segment["label"]
        heading_segment["label"] = HEADING_LABEL
    return len(heading_segments)
