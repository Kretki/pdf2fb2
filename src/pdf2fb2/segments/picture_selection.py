"""Picks the picture block for an off-size region and puts it into the page in place of the text it covers.

The layout model proposes boxes for the page again at a low threshold. The most probable picture box that
covers at least half of the region wins, provided it is more probable than the text segment it would replace.
The picture then takes in:

* every segment it covers for at least half of the segment's area;
* body-text segments right above or below it (gap up to 3 usual glyph heights, overlapping horizontally)
  whose lettering does not fit the usual font size, e.g. the small print at the foot of a poster.
"""

from dataclasses import dataclass

from pdf2fb2.segments.font_statistics import GLYPH_HEIGHT_FIELD, FontStatistics, OffSizeRegion, fits_usual_font
from pdf2fb2.segments.label_roles import HANDWRITTEN_LABEL, IMAGE_LABEL, PICTURE_LABELS
from pdf2fb2.segments.position_statistics import BODY_TEXT_ROLE, position_role
from pdf2fb2.segments.segment_geometry import box_area, box_intersection, union_box

REGION_COVERAGE_RATIO = 0.5
SEGMENT_COVERAGE_RATIO = 0.5
HORIZONTAL_OVERLAP_RATIO = 0.5
ADJACENT_GAP_FACTOR = 3.0  # in usual glyph heights
RECOVERED_FIELD = "recovered_from"  # picture segment field: labels of the segments it replaced


@dataclass(frozen=True)
class PictureChoice:
    """The picture block chosen for one region.

    Attributes:
        picture_box: [x1, y1, x2, y2] of the picture in page pixels (grown by the segments it takes in).
        picture_score: Layout score of the picture proposal.
        absorbed_indexes: Indexes of the page segments the picture replaces.
    """

    picture_box: list[float]
    picture_score: float
    absorbed_indexes: frozenset[int]


def best_picture_proposal(proposed_boxes: list[dict], off_size_region: OffSizeRegion) -> dict | None:
    """Most probable picture proposal covering a region, if it beats the text it would replace.

    Args:
        proposed_boxes: Layout boxes ``{"label", "bbox", "score"}`` proposed for the region's page.
        off_size_region: The region.

    Returns:
        The proposal, or None when no picture covers half of the region with a score above
        ``off_size_region.competing_score``.
    """
    region_area = box_area(off_size_region.region_box) or 1.0
    picture_proposals = [proposed_box for proposed_box in proposed_boxes
                         if proposed_box["label"] in PICTURE_LABELS
                         and proposed_box["score"] > off_size_region.competing_score
                         and box_intersection(proposed_box["bbox"], off_size_region.region_box)
                         >= REGION_COVERAGE_RATIO * region_area]
    return max(picture_proposals, key=lambda proposed_box: proposed_box["score"], default=None)


def is_covered(segment_box: list[float], picture_box: list[float]) -> bool:
    """Tell whether a picture box covers at least half of a segment.

    Args:
        segment_box: [x1, y1, x2, y2] of the segment.
        picture_box: [x1, y1, x2, y2] of the picture.

    Returns:
        True if the intersection is at least ``SEGMENT_COVERAGE_RATIO`` of the segment area.
    """
    return box_intersection(segment_box, picture_box) >= SEGMENT_COVERAGE_RATIO * (box_area(segment_box) or 1.0)


def is_adjacent(segment_box: list[float], picture_box: list[float], maximum_gap: float) -> bool:
    """Tell whether a segment sits right above or below a picture.

    Args:
        segment_box: [x1, y1, x2, y2] of the segment.
        picture_box: [x1, y1, x2, y2] of the picture.
        maximum_gap: Largest vertical gap in pixels (>= 0).

    Returns:
        True if the vertical gap is at most ``maximum_gap`` and the boxes overlap horizontally for at least
        ``HORIZONTAL_OVERLAP_RATIO`` of the segment width.
    """
    vertical_gap = max(segment_box[1] - picture_box[3], picture_box[1] - segment_box[3])
    horizontal_overlap = min(segment_box[2], picture_box[2]) - max(segment_box[0], picture_box[0])
    return vertical_gap <= maximum_gap and horizontal_overlap >= HORIZONTAL_OVERLAP_RATIO * (
        segment_box[2] - segment_box[0])


def belongs_to_picture(page_segment: dict, picture_box: list[float], font_statistics: FontStatistics) -> bool:
    """Tell whether a segment is part of a picture block.

    Args:
        page_segment: Segment of the picture's page.
        picture_box: [x1, y1, x2, y2] of the picture so far.
        font_statistics: Usual font size of the book.

    Returns:
        True for covered segments, and for adjacent body text whose lettering does not fit the usual font;
        handwritten segments are never taken in.
    """
    if page_segment["label"] == HANDWRITTEN_LABEL:
        return False
    if is_covered(page_segment["bbox"], picture_box):
        return True
    glyph_height = page_segment.get(GLYPH_HEIGHT_FIELD)
    return (glyph_height is not None and position_role(page_segment["label"]) == BODY_TEXT_ROLE
            and not fits_usual_font(glyph_height, font_statistics)
            and is_adjacent(page_segment["bbox"], picture_box, ADJACENT_GAP_FACTOR * font_statistics.typical_height))


def grow_picture(page_segments: list[dict], picture_box: list[float],
                 font_statistics: FontStatistics) -> tuple[list[float], frozenset[int]]:
    """Take in the segments that belong to a picture, until no more do.

    Args:
        page_segments: Segments of the picture's page.
        picture_box: [x1, y1, x2, y2] of the picture proposal (with the region).
        font_statistics: Usual font size of the book.

    Returns:
        The grown picture box and the indexes of the taken-in segments.
    """
    absorbed_indexes: set[int] = set()
    while True:
        new_indexes = {segment_index for segment_index, page_segment in enumerate(page_segments)
                       if segment_index not in absorbed_indexes
                       and belongs_to_picture(page_segment, picture_box, font_statistics)}
        if not new_indexes:
            return picture_box, frozenset(absorbed_indexes)
        absorbed_indexes |= new_indexes
        picture_box = union_box([picture_box, *(page_segments[segment_index]["bbox"]
                                                for segment_index in new_indexes)])


def choose_picture(off_size_region: OffSizeRegion, proposed_boxes: list[dict],
                   font_statistics: FontStatistics) -> PictureChoice | None:
    """Choose the picture block of a region and the segments it replaces.

    Usage:
        Call for every off-size region with the low-threshold layout proposals of its page.

    Args:
        off_size_region: The region.
        proposed_boxes: Layout boxes proposed for the region's page.
        font_statistics: Usual font size of the book.

    Returns:
        The choice, or None when the layout model proposes no fitting picture.
    """
    picture_proposal = best_picture_proposal(proposed_boxes, off_size_region)
    if picture_proposal is None:
        return None
    picture_box, absorbed_indexes = grow_picture(off_size_region.page_record["segments"],
                                                 union_box([picture_proposal["bbox"], off_size_region.region_box]),
                                                 font_statistics)
    return PictureChoice(picture_box, picture_proposal["score"], absorbed_indexes)


def picture_insert_index(remaining_segments: list[dict], picture_box: list[float]) -> int:
    """Reading-order position of a new picture: before the first segment that starts below its top.

    Args:
        remaining_segments: Page segments in reading order, without the replaced ones.
        picture_box: [x1, y1, x2, y2] of the picture.

    Returns:
        Insertion index (``len(remaining_segments)`` when every segment starts above the picture).
    """
    return next((segment_index for segment_index, page_segment in enumerate(remaining_segments)
                 if page_segment["bbox"][1] >= picture_box[1]), len(remaining_segments))


def apply_picture(page_record: dict, picture_choice: PictureChoice) -> None:
    """Replace the absorbed segments of a page by the chosen picture.

    Args:
        page_record: Page record (mutated: ``segments`` replaced, ``order`` renumbered 0, 1, 2, ...).
        picture_choice: Picture and the indexes of the segments it replaces.

    Returns:
        None.
    """
    page_segments = page_record["segments"]
    remaining_segments = [page_segment for segment_index, page_segment in enumerate(page_segments)
                          if segment_index not in picture_choice.absorbed_indexes]
    picture_segment = {"label": IMAGE_LABEL, "bbox": picture_choice.picture_box, "score": picture_choice.picture_score,
                       "order": None, RECOVERED_FIELD: [page_segments[segment_index]["label"]
                                                        for segment_index in sorted(picture_choice.absorbed_indexes)]}
    remaining_segments.insert(picture_insert_index(remaining_segments, picture_choice.picture_box), picture_segment)
    for order_index, page_segment in enumerate(remaining_segments):
        page_segment["order"] = order_index
    page_record["segments"] = remaining_segments
