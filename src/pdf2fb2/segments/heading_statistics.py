"""Statistical heading check: short, centred, sparsely placed "paragraphs" on near-empty pages are titles.

Title pages, half-titles and section openings carry a few short lines in the middle of an almost empty page,
and the layout model often labels them plain text. A ``text`` segment is relabelled as a heading when:

* its page is sparse: text covers at most 30% of the median text coverage of the book's pages;
* it is short: its text is not longer than the 10th percentile of the book's paragraphs;
* it is centred on the body-text column of its page group (same page parity);
* it stands apart (gap to the nearest text at least 5 usual glyph heights) or is set larger than usual.

The lines of a list on a sparse page (e.g. a list of illustrations) are short and centred too, but they follow
each other closely in the usual font, so they stay text.
"""

import re
import statistics
from collections import defaultdict
from dataclasses import dataclass

from pdf2fb2.segments.font_statistics import GLYPH_HEIGHT_FIELD, FontStatistics, collect_font_statistics
from pdf2fb2.segments.label_roles import BODY_TEXT_LABEL, PAGE_NUMBER_LABELS, requires_ocr
from pdf2fb2.segments.position_statistics import center_ratio
from pdf2fb2.segments.segment_geometry import box_area

HEADING_LABEL = "paragraph_title"
SPARSE_COVERAGE_FACTOR = 0.3
SHORT_TEXT_QUANTILE = 0  # index of the 10th percentile among the decile cut points
CENTER_TOLERANCE_RATIO = 0.06  # fraction of the page width (scans shift title pages by a few percent)
ISOLATION_GAP_FACTOR = 5.0  # in usual glyph heights
FALLBACK_GLYPH_HEIGHT = 14.0  # pixels, when the glyph heights were not measured
MINIMUM_REFERENCE_PARAGRAPHS = 10
MARKUP_TAG_PATTERN = re.compile(r"</?\w+>")


@dataclass(frozen=True)
class BookTextStatistics:
    """Statistics of the book's text that headings are compared with.

    Attributes:
        median_coverage: Median share of the page area covered by text, over pages with text.
        short_text_length: 10th percentile of the paragraph text lengths, in characters.
        column_centers: Median horizontal center of the paragraphs per page parity, as a fraction of the width.
        font_statistics: Usual font size, or None when the glyph heights were not measured.
    """

    median_coverage: float
    short_text_length: float
    column_centers: dict[int, float]
    font_statistics: FontStatistics | None


def visible_length(page_segment: dict) -> int:
    """Length of a segment's text without markup tags.

    Args:
        page_segment: OCR'd segment (``text`` may carry ``<b>``/``<i>`` style tags).

    Returns:
        Number of visible characters, surrounding spaces excluded.
    """
    return len(MARKUP_TAG_PATTERN.sub("", page_segment.get("text") or "").strip())


def is_text_block(page_segment: dict) -> bool:
    """Tell whether a segment is OCR'd text that occupies the page (page numbers excluded).

    Args:
        page_segment: Segment.

    Returns:
        True for OCR'd segments other than page numbers.
    """
    return requires_ocr(page_segment["label"]) and page_segment["label"] not in PAGE_NUMBER_LABELS


def text_coverage(page_record: dict) -> float:
    """Share of the page area covered by text blocks.

    Args:
        page_record: Page record.

    Returns:
        Covered area / page area (0 for pages of zero size).
    """
    page_area = float(page_record["size"][0]) * float(page_record["size"][1])
    if page_area <= 0:
        return 0.0
    return sum(box_area(page_segment["bbox"]) for page_segment in page_record["segments"]
               if is_text_block(page_segment)) / page_area


def column_centers(page_records: list[dict]) -> dict[int, float]:
    """Median horizontal center of the paragraphs per page parity.

    Args:
        page_records: Page records of the book.

    Returns:
        Mapping page parity (0/1) -> center as a fraction of the page width.
    """
    centers_by_parity: dict[int, list[float]] = defaultdict(list)
    for page_record in page_records:
        page_width = float(page_record["size"][0])
        for page_segment in page_record["segments"]:
            if page_segment["label"] == BODY_TEXT_LABEL and page_width > 0:
                centers_by_parity[page_record["page"] % 2].append(center_ratio(page_segment["bbox"], page_width))
    return {page_parity: statistics.median(center_values) for page_parity, center_values in centers_by_parity.items()}


def collect_text_statistics(page_records: list[dict]) -> BookTextStatistics | None:
    """Text statistics of the whole book.

    Usage:
        Call on the OCR'd page records.

    Args:
        page_records: OCR'd page records (``glyph_height`` used when stored).

    Returns:
        The statistics, or None when the book has fewer than ``MINIMUM_REFERENCE_PARAGRAPHS`` paragraphs.
    """
    text_lengths = [visible_length(page_segment) for page_record in page_records
                    for page_segment in page_record["segments"] if page_segment["label"] == BODY_TEXT_LABEL]
    if len(text_lengths) < MINIMUM_REFERENCE_PARAGRAPHS:
        return None
    page_coverages = [text_coverage(page_record) for page_record in page_records]
    return BookTextStatistics(statistics.median([coverage for coverage in page_coverages if coverage > 0]),
                              statistics.quantiles(text_lengths, n=10)[SHORT_TEXT_QUANTILE],
                              column_centers(page_records), collect_font_statistics(page_records))


def vertical_gap(first_box: list[float], second_box: list[float]) -> float:
    """Vertical distance between two boxes.

    Args:
        first_box: [x1, y1, x2, y2] in pixels.
        second_box: [x1, y1, x2, y2] in pixels.

    Returns:
        Gap in pixels; 0 when the boxes overlap vertically.
    """
    return max(0.0, second_box[1] - first_box[3], first_box[1] - second_box[3])


def stands_apart(page_segment: dict, page_record: dict, text_statistics: BookTextStatistics) -> bool:
    """Tell whether a segment is far from every other text block of its page, or set larger than usual.

    Args:
        page_segment: Candidate segment.
        page_record: Its page record.
        text_statistics: Statistics of the book.

    Returns:
        True when the nearest text is at least ``ISOLATION_GAP_FACTOR`` usual glyph heights away, or when the
        segment's glyphs are taller than the normal range of the book's paragraphs.
    """
    font_statistics = text_statistics.font_statistics
    glyph_height = page_segment.get(GLYPH_HEIGHT_FIELD)
    if font_statistics is not None and glyph_height is not None and glyph_height > font_statistics.upper_height:
        return True
    usual_height = font_statistics.typical_height if font_statistics is not None else FALLBACK_GLYPH_HEIGHT
    other_gaps = [vertical_gap(page_segment["bbox"], other_segment["bbox"])
                  for other_segment in page_record["segments"]
                  if other_segment is not page_segment and is_text_block(other_segment)]
    return min(other_gaps, default=float("inf")) >= ISOLATION_GAP_FACTOR * usual_height


def is_sparse_heading(page_segment: dict, page_record: dict, text_statistics: BookTextStatistics) -> bool:
    """Tell whether a text segment is a heading set alone on a near-empty page.

    Args:
        page_segment: Segment of the page.
        page_record: Its page record.
        text_statistics: Statistics of the book.

    Returns:
        True for short, centred ``text`` segments on sparse pages that stand apart or are set larger.
    """
    page_width = float(page_record["size"][0])
    column_center = text_statistics.column_centers.get(page_record["page"] % 2)
    if page_segment["label"] != BODY_TEXT_LABEL or column_center is None or page_width <= 0:
        return False
    return (visible_length(page_segment) <= text_statistics.short_text_length
            and abs(center_ratio(page_segment["bbox"], page_width) - column_center) <= CENTER_TOLERANCE_RATIO
            and stands_apart(page_segment, page_record, text_statistics))


def mark_sparse_headings(page_records: list[dict]) -> int:
    """Relabel the short, centred, sparsely placed paragraphs of near-empty pages as headings.

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
    sparse_limit = SPARSE_COVERAGE_FACTOR * text_statistics.median_coverage
    heading_segments = [page_segment for page_record in page_records if text_coverage(page_record) <= sparse_limit
                        for page_segment in page_record["segments"]
                        if is_sparse_heading(page_segment, page_record, text_statistics)]
    for heading_segment in heading_segments:
        heading_segment["original_label"] = heading_segment["label"]
        heading_segment["label"] = HEADING_LABEL
    return len(heading_segments)
