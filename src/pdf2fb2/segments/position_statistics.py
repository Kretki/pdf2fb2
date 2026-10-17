"""Statistical layout check: finds segments whose horizontal position differs from the book's typical one.

The body text of a book shares its column boundaries and headings sit at about one horizontal position,
while human-written notes are placed freely. A segment whose position lies outside the sigma interval of
its group (same role, same page parity, since facing pages have mirrored margins) is an outlier.
Positions are fractions of the page width, so pages rendered at different sizes are comparable.
"""

import statistics
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

from pdf2fb2.segments.label_roles import (
    CAPTION_LABELS,
    DOCUMENT_TITLE_LABEL,
    DROPPED_LABELS,
    HEADING_LABELS,
    PAGE_NUMBER_LABELS,
    TABLE_LABEL,
    requires_ocr,
)

OUTLIER_SIGMA_LIMIT = 2.0
MINIMUM_GROUP_SIZE = 5
# Deviation floor (fraction of the page width): a perfectly regular layout would otherwise flag 1-pixel jitter.
MINIMUM_DEVIATION_RATIO = 0.01
BODY_TEXT_ROLE = "body_text"
HEADING_ROLE = "heading"
UNCOMPARED_LABELS = CAPTION_LABELS | PAGE_NUMBER_LABELS | DROPPED_LABELS | {DOCUMENT_TITLE_LABEL, TABLE_LABEL}

PositionFeature = Callable[[list[float], float], float]
GroupKey = tuple[str, int]


def left_edge_ratio(bounding_box: list[float], page_width: float) -> float:
    """Left edge of a box as a fraction of the page width.

    Args:
        bounding_box: [x1, y1, x2, y2] in page pixels.
        page_width: Page image width in pixels (> 0).

    Returns:
        x1 / page_width, normally 0..1.
    """
    return bounding_box[0] / page_width


def right_edge_ratio(bounding_box: list[float], page_width: float) -> float:
    """Right edge of a box as a fraction of the page width.

    Args:
        bounding_box: [x1, y1, x2, y2] in page pixels.
        page_width: Page image width in pixels (> 0).

    Returns:
        x2 / page_width, normally 0..1.
    """
    return bounding_box[2] / page_width


def center_ratio(bounding_box: list[float], page_width: float) -> float:
    """Horizontal center of a box as a fraction of the page width.

    Args:
        bounding_box: [x1, y1, x2, y2] in page pixels.
        page_width: Page image width in pixels (> 0).

    Returns:
        (x1 + x2) / 2 / page_width, normally 0..1.
    """
    return (bounding_box[0] + bounding_box[2]) / 2 / page_width


# Body text is compared by both column edges, headings by their placement (centered or flush).
ROLE_FEATURES: dict[str, tuple[PositionFeature, ...]] = {
    BODY_TEXT_ROLE: (left_edge_ratio, right_edge_ratio),
    HEADING_ROLE: (center_ratio,),
}


def position_role(segment_label: str) -> str | None:
    """Tell which statistical group a segment label belongs to.

    Usage:
        Call for every segment before collecting its position features.

    Args:
        segment_label: PP-DocLayoutV3 label of the segment.

    Returns:
        ``HEADING_ROLE`` for headings, ``BODY_TEXT_ROLE`` for other OCR'd text (unknown labels included),
        None for pictures, tables, captions, page numbers, the title and running heads (not compared).
    """
    if segment_label in HEADING_LABELS:
        return HEADING_ROLE
    if requires_ocr(segment_label) and segment_label not in UNCOMPARED_LABELS:
        return BODY_TEXT_ROLE
    return None


@dataclass(frozen=True)
class PositionedSegment:
    """A compared segment with its position features.

    Attributes:
        page_record: Page record the segment belongs to.
        page_segment: The segment itself (same object as in the page record).
        group_key: (role, page parity 0/1) of the statistical group.
        feature_values: Position features of the role, as fractions of the page width.
    """

    page_record: dict
    page_segment: dict
    group_key: GroupKey
    feature_values: tuple[float, ...]


@dataclass(frozen=True)
class FeatureSpread:
    """Mean and deviation of every position feature of one group.

    Attributes:
        mean_values: Mean of each feature.
        deviation_values: Population standard deviation of each feature, at least ``MINIMUM_DEVIATION_RATIO``.
    """

    mean_values: tuple[float, ...]
    deviation_values: tuple[float, ...]


def positioned_page_segments(page_record: dict) -> list[PositionedSegment]:
    """Position features of the compared segments of one page.

    Args:
        page_record: Segmented page record with ``page``, ``size`` and ``segments``.

    Returns:
        One entry per segment that has a position role; empty for pages of zero width.
    """
    page_width = float(page_record["size"][0])
    if page_width <= 0:
        return []
    page_parity = page_record["page"] % 2
    positioned_segments = []
    for page_segment in page_record["segments"]:
        segment_role = position_role(page_segment["label"])
        if segment_role is not None:
            feature_values = tuple(position_feature(page_segment["bbox"], page_width)
                                   for position_feature in ROLE_FEATURES[segment_role])
            positioned_segments.append(PositionedSegment(page_record, page_segment, (segment_role, page_parity),
                                                         feature_values))
    return positioned_segments


def compute_spread(feature_rows: list[tuple[float, ...]]) -> FeatureSpread:
    """Mean and floored standard deviation of every feature column.

    Args:
        feature_rows: Feature tuples of one group (at least one row, all of the same length).

    Returns:
        Spread of the group.
    """
    feature_columns = list(zip(*feature_rows, strict=True))
    return FeatureSpread(
        tuple(statistics.fmean(feature_column) for feature_column in feature_columns),
        tuple(max(statistics.pstdev(feature_column), MINIMUM_DEVIATION_RATIO) for feature_column in feature_columns))


def group_spreads(positioned_segments: list[PositionedSegment]) -> dict[GroupKey, FeatureSpread]:
    """Spread of every group with enough segments for statistics.

    Args:
        positioned_segments: Compared segments of the whole book.

    Returns:
        Mapping group key -> spread; groups with fewer than ``MINIMUM_GROUP_SIZE`` segments are absent.
    """
    grouped_rows: dict[GroupKey, list[tuple[float, ...]]] = defaultdict(list)
    for positioned_segment in positioned_segments:
        grouped_rows[positioned_segment.group_key].append(positioned_segment.feature_values)
    return {group_key: compute_spread(feature_rows) for group_key, feature_rows in grouped_rows.items()
            if len(feature_rows) >= MINIMUM_GROUP_SIZE}


def lies_outside(positioned_segment: PositionedSegment, feature_spread: FeatureSpread, sigma_limit: float) -> bool:
    """Tell whether any position feature of a segment is outside the sigma interval of its group.

    Args:
        positioned_segment: Segment to test.
        feature_spread: Spread of the segment's group.
        sigma_limit: Interval half-width in standard deviations (> 0).

    Returns:
        True if ``|value - mean| > sigma_limit * deviation`` for at least one feature.
    """
    return any(abs(feature_value - mean_value) > sigma_limit * deviation_value
               for feature_value, mean_value, deviation_value in zip(
                   positioned_segment.feature_values, feature_spread.mean_values, feature_spread.deviation_values,
                   strict=True))


def find_position_outliers(page_records: list[dict],
                           sigma_limit: float = OUTLIER_SIGMA_LIMIT) -> list[PositionedSegment]:
    """Find the segments whose horizontal position is statistically unusual for the book.

    Usage:
        Call on the segmented page records, before reader-mark cleaning and OCR.

    Args:
        page_records: Segmented page records of the whole book.
        sigma_limit: Interval half-width in standard deviations (default 2.0).

    Returns:
        Outlying segments in page and reading order (groups too small for statistics are never flagged).
    """
    positioned_segments = [positioned_segment for page_record in page_records
                           for positioned_segment in positioned_page_segments(page_record)]
    spread_by_group = group_spreads(positioned_segments)
    return [positioned_segment for positioned_segment in positioned_segments
            if positioned_segment.group_key in spread_by_group
            and lies_outside(positioned_segment, spread_by_group[positioned_segment.group_key], sigma_limit)]
