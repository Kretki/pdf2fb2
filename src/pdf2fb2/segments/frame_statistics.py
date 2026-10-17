"""Statistical layout check: finds text segments that stick out above or below the book's printed frame.

Every printed page puts its content (running head, text, page number) inside about the same vertical band,
while a reader's note can be written anywhere, e.g. in the top margin above the running head. The frame of a
group of pages (same page parity) runs from the pages' content tops to their content bottoms, after the 5% most
extreme pages on each side (at least one) are discarded, so a single annotated page never defines the frame;
a text segment protruding past it by more than the tolerance is outside the book. Positions are fractions
of the page height, so pages rendered at different sizes are comparable.
Pictures do not define the frame (they may bleed to the page edge) and captions are not tested (they follow
their pictures).
"""

from collections import defaultdict
from dataclasses import dataclass

from pdf2fb2.segments.label_roles import CAPTION_LABELS, HANDWRITTEN_LABEL, PICTURE_LABELS, requires_ocr
from pdf2fb2.segments.position_statistics import MINIMUM_GROUP_SIZE, PositionedSegment

# Protrusion past the frame (fraction of the page height, about one line of body text) that is not printed.
FRAME_TOLERANCE_RATIO = 0.02
FRAME_ROLE = "frame"
FRAME_TRIM_SHARE = 0.05  # share of the most extreme pages ignored on each side of the frame (at least one page)
NON_FRAME_LABELS = PICTURE_LABELS | {HANDWRITTEN_LABEL}


@dataclass(frozen=True)
class PageFrame:
    """Vertical band of the printed content of a group of pages.

    Attributes:
        frame_top: Upper frame edge as a fraction of the page height, 0..1.
        frame_bottom: Lower frame edge as a fraction of the page height, 0..1.
    """

    frame_top: float
    frame_bottom: float


def content_extent(page_record: dict) -> tuple[float, float] | None:
    """Vertical extent of the printed (non-picture) content of one page.

    Args:
        page_record: Segmented page record with ``size`` and ``segments``.

    Returns:
        (top, bottom) as fractions of the page height, or None for a page without such segments
        or of zero height.
    """
    page_height = float(page_record["size"][1])
    frame_segments = [page_segment for page_segment in page_record["segments"]
                      if page_segment["label"] not in NON_FRAME_LABELS]
    if page_height <= 0 or not frame_segments:
        return None
    return (min(page_segment["bbox"][1] for page_segment in frame_segments) / page_height,
            max(page_segment["bbox"][3] for page_segment in frame_segments) / page_height)


def trimmed_minimum(extent_values: list[float]) -> float:
    """Smallest value after the lowest 5% (at least one value) are discarded.

    Args:
        extent_values: Values of one page group (at least two).

    Returns:
        The value at rank ``max(1, int(FRAME_TRIM_SHARE * count))`` in ascending order.
    """
    return sorted(extent_values)[max(1, int(FRAME_TRIM_SHARE * len(extent_values)))]


def group_frames(page_records: list[dict]) -> dict[int, PageFrame]:
    """Printed frame of every page-parity group with enough pages for statistics.

    Args:
        page_records: Segmented page records of the whole book.

    Returns:
        Mapping page parity (0/1) -> frame; groups with fewer than ``MINIMUM_GROUP_SIZE`` pages are absent.
    """
    extents_by_parity: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for page_record in page_records:
        page_extent = content_extent(page_record)
        if page_extent is not None:
            extents_by_parity[page_record["page"] % 2].append(page_extent)
    return {page_parity: PageFrame(trimmed_minimum([page_extent[0] for page_extent in page_extents]),
                                   -trimmed_minimum([-page_extent[1] for page_extent in page_extents]))
            for page_parity, page_extents in extents_by_parity.items() if len(page_extents) >= MINIMUM_GROUP_SIZE}


def frame_protrusion(page_segment: dict, page_height: float, page_frame: PageFrame) -> tuple[float, float]:
    """How far a segment sticks out above and below the frame.

    Args:
        page_segment: Segment with ``bbox`` in page pixels.
        page_height: Page image height in pixels (> 0).
        page_frame: Frame of the segment's page group.

    Returns:
        (above, below) protrusion as fractions of the page height; negative when inside the frame.
    """
    return (page_frame.frame_top - page_segment["bbox"][1] / page_height,
            page_segment["bbox"][3] / page_height - page_frame.frame_bottom)


def is_frame_tested(segment_label: str) -> bool:
    """Tell whether a segment label is tested against the printed frame.

    Args:
        segment_label: Layout label of the segment.

    Returns:
        True for OCR'd text segments except captions.
    """
    return requires_ocr(segment_label) and segment_label not in CAPTION_LABELS


def find_frame_outliers(page_records: list[dict],
                        tolerance_ratio: float = FRAME_TOLERANCE_RATIO) -> list[PositionedSegment]:
    """Find the text segments that protrude above or below the printed frame of the book.

    Usage:
        Call on the segmented page records, before reader-mark cleaning and OCR. Nothing printed in the
        book lies there, so the outliers are reader notes.

    Args:
        page_records: Segmented page records of the whole book.
        tolerance_ratio: Allowed protrusion as a fraction of the page height (>= 0, default 0.02).

    Returns:
        Outliers in page and reading order, with group key ("frame", parity) and the (above, below)
        protrusions as features; page groups too small for statistics are never tested.
    """
    frame_by_parity = group_frames(page_records)
    return [frame_outlier for page_record in page_records if page_record["page"] % 2 in frame_by_parity
            for frame_outlier in page_frame_outliers(page_record, frame_by_parity[page_record["page"] % 2],
                                                     tolerance_ratio)]


def page_frame_outliers(page_record: dict, page_frame: PageFrame, tolerance_ratio: float) -> list[PositionedSegment]:
    """Text segments of one page that protrude past the frame by more than the tolerance.

    Args:
        page_record: Segmented page record.
        page_frame: Frame of the page's parity group.
        tolerance_ratio: Allowed protrusion as a fraction of the page height (>= 0).

    Returns:
        Outliers in reading order; empty for pages of zero height.
    """
    page_height = float(page_record["size"][1])
    if page_height <= 0:
        return []
    group_key = (FRAME_ROLE, page_record["page"] % 2)
    tested_segments = [page_segment for page_segment in page_record["segments"]
                       if is_frame_tested(page_segment["label"])]
    protrusions = [(page_segment, frame_protrusion(page_segment, page_height, page_frame))
                   for page_segment in tested_segments]
    return [PositionedSegment(page_record, page_segment, group_key, protrusion_values)
            for page_segment, protrusion_values in protrusions if max(protrusion_values) > tolerance_ratio]
