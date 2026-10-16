"""Pure box geometry: overlap removal and reading order of page segments.

A segment is a dict with at least ``label`` (str), ``bbox`` ([x1, y1, x2, y2] in
page-image pixels) and ``order`` (int | None).
"""

from pdf2fb2.segments.label_roles import DROPPED_LABELS

NESTED_OVERLAP_RATIO = 0.8
FULL_WIDTH_RATIO = 0.6


def box_area(bounding_box: list[float]) -> float:
    """Return the area of a box, 0 for degenerate boxes.

    Args:
        bounding_box: [x1, y1, x2, y2] in pixels.

    Returns:
        Area in square pixels (>= 0).
    """
    return max(0.0, bounding_box[2] - bounding_box[0]) * max(0.0, bounding_box[3] - bounding_box[1])


def box_intersection(first_box: list[float], second_box: list[float]) -> float:
    """Return the intersection area of two boxes.

    Args:
        first_box: [x1, y1, x2, y2] in pixels.
        second_box: [x1, y1, x2, y2] in pixels.

    Returns:
        Overlap area in square pixels (>= 0).
    """
    return box_area([max(first_box[0], second_box[0]), max(first_box[1], second_box[1]),
                     min(first_box[2], second_box[2]), min(first_box[3], second_box[3])])


def _is_nested(candidate: dict, kept_segments: list[dict]) -> bool:
    """Tell whether a segment lies mostly inside an already kept one.

    Args:
        candidate: Segment being considered.
        kept_segments: Larger segments kept so far.

    Returns:
        True if more than 80% of the candidate's area is inside a kept segment.
    """
    candidate_area = box_area(candidate["bbox"]) or 1.0
    return any(box_intersection(candidate["bbox"], kept_segment["bbox"]) / candidate_area > NESTED_OVERLAP_RATIO
               for kept_segment in kept_segments)


def remove_nested_segments(page_segments: list[dict]) -> list[dict]:
    """Drop inline formulas and boxes that are mostly inside a bigger box.

    Usage:
        Call on the raw detector output of one page.

    Args:
        page_segments: Segments of one page.

    Returns:
        New list of kept segments, largest first.
    """
    candidates = sorted((page_segment for page_segment in page_segments if page_segment["label"] not in DROPPED_LABELS),
                        key=lambda page_segment: -box_area(page_segment["bbox"]))
    kept_segments: list[dict] = []
    for candidate in candidates:
        if not _is_nested(candidate, kept_segments):
            kept_segments.append(candidate)
    return kept_segments


def _segment_center_x(page_segment: dict) -> float:
    """Return the horizontal center of a segment in pixels.

    Args:
        page_segment: Segment with ``bbox``.

    Returns:
        (x1 + x2) / 2.
    """
    return (page_segment["bbox"][0] + page_segment["bbox"][2]) / 2


def _split_column_band(column_band: list[dict], page_width: float) -> list[dict]:
    """Order a band of column segments: left column first, then right column.

    Args:
        column_band: Segments between two full-width blocks, sorted top to bottom.
        page_width: Page image width in pixels.

    Returns:
        Segments of the left half followed by those of the right half.
    """
    left_column = [page_segment for page_segment in column_band if _segment_center_x(page_segment) < page_width / 2]
    right_column = [page_segment for page_segment in column_band if _segment_center_x(page_segment) >= page_width / 2]
    return left_column + right_column


def _order_by_columns(page_segments: list[dict], page_width: float) -> list[dict]:
    """Column-aware top-to-bottom reading order heuristic.

    Args:
        page_segments: Segments of one page.
        page_width: Page image width in pixels.

    Returns:
        Segments in reading order; full-width blocks separate column bands.
    """
    top_down_segments = sorted(page_segments,
                               key=lambda page_segment: (page_segment["bbox"][1], page_segment["bbox"][0]))
    ordered_segments: list[dict] = []
    column_band: list[dict] = []
    for page_segment in top_down_segments:
        if page_segment["bbox"][2] - page_segment["bbox"][0] > FULL_WIDTH_RATIO * page_width:
            ordered_segments.extend(_split_column_band(column_band, page_width))
            column_band = []
            ordered_segments.append(page_segment)
        else:
            column_band.append(page_segment)
    ordered_segments.extend(_split_column_band(column_band, page_width))
    return ordered_segments


def order_segments_for_reading(page_segments: list[dict], page_width: float) -> list[dict]:
    """Sort segments in reading order and renumber their ``order`` field.

    Usage:
        Call after ``remove_nested_segments``. Uses the model's order when every
        segment has one, else a column-aware heuristic.

    Args:
        page_segments: Segments of one page (mutated: ``order`` is rewritten).
        page_width: Page image width in pixels.

    Returns:
        Segments in reading order with ``order`` = 0, 1, 2, ...
    """
    if page_segments and all(isinstance(page_segment.get("order"), (int, float)) for page_segment in page_segments):
        ordered_segments = sorted(page_segments, key=lambda page_segment: page_segment["order"])
    else:
        ordered_segments = _order_by_columns(page_segments, page_width)
    for order_index, page_segment in enumerate(ordered_segments):
        page_segment["order"] = order_index
    return ordered_segments
