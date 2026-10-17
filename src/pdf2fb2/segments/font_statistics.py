"""Statistical font check: finds page regions whose lettering does not fit the usual font size of the book.

Glyph height (median height of the letter-like ink components of a segment, see
``segments.glyph_measurement``) stands for the font size. The usual font size is the median glyph height of
the plain paragraphs of the whole book, and its normal range spans their 5th..95th percentile. Two things
make a page region suspicious, as both are typical of a picture full of lettering (a poster, a facsimile)
that the layout model has taken for text:

* a body-text segment whose glyph height is outside 0.5x..2x of the usual one;
* many letter-like ink components outside every segment that are at least 2.5x the usual height, inside the
  printed frame of the book (large lettering outside it is a reader's note, see ``segments.frame_statistics``).
"""

import statistics
from dataclasses import dataclass

from pdf2fb2.segments.frame_statistics import FRAME_TOLERANCE_RATIO, PageFrame, group_frames
from pdf2fb2.segments.label_roles import BODY_TEXT_LABEL
from pdf2fb2.segments.position_statistics import BODY_TEXT_ROLE, position_role
from pdf2fb2.segments.segment_geometry import union_box

GLYPH_HEIGHT_FIELD = "glyph_height"  # segment field: median glyph height in pixels, set by the picture recovery
MINIMUM_REFERENCE_SEGMENTS = 5
FONT_QUANTILE_COUNT = 20  # cut points every 5%: the normal range drops the 5% most extreme paragraphs per side
OFF_SIZE_LOWER_FACTOR = 0.5
OFF_SIZE_UPPER_FACTOR = 2.0
LARGE_GLYPH_FACTOR = 2.5
MINIMUM_LARGE_GLYPHS = 8


@dataclass(frozen=True)
class PageGlyphs:
    """Glyph measurements of one page.

    Attributes:
        segment_heights: Median glyph height in pixels of every segment, aligned with the page's segments;
            None where the segment has too few glyphs to measure.
        uncovered_glyphs: [x1, y1, x2, y2] pixel boxes of letter-like ink components outside every segment.
    """

    segment_heights: tuple[float | None, ...]
    uncovered_glyphs: tuple[tuple[int, int, int, int], ...]


@dataclass(frozen=True)
class FontStatistics:
    """Usual font size of a book, as glyph heights in pixels.

    Attributes:
        typical_height: Median glyph height of the plain paragraphs (> 0).
        lower_height: 5th percentile of the paragraph glyph heights.
        upper_height: 95th percentile of the paragraph glyph heights.
        paragraph_count: Number of measured paragraphs.
    """

    typical_height: float
    lower_height: float
    upper_height: float
    paragraph_count: int


@dataclass(frozen=True)
class OffSizeRegion:
    """A page region whose lettering does not fit the usual font size.

    Attributes:
        page_record: Page record the region belongs to.
        region_box: [x1, y1, x2, y2] of the region in page pixels.
        region_reason: ``off-size text`` (a segment) or ``large lettering`` (glyphs outside every segment).
        competing_score: Layout score of the text segment a picture would replace (0 for large lettering):
            a picture is taken only when the layout model finds it more probable.
    """

    page_record: dict
    region_box: list[float]
    region_reason: str
    competing_score: float


def store_glyph_heights(page_record: dict, page_glyphs: PageGlyphs) -> None:
    """Write the measured glyph heights into the segments of a page.

    Args:
        page_record: Page record (mutated: ``glyph_height`` set on every measured segment).
        page_glyphs: Measurements of the same page, aligned with its segments.

    Returns:
        None.
    """
    for page_segment, glyph_height in zip(page_record["segments"], page_glyphs.segment_heights, strict=True):
        if glyph_height is not None:
            page_segment[GLYPH_HEIGHT_FIELD] = glyph_height


def collect_font_statistics(page_records: list[dict]) -> FontStatistics | None:
    """Usual font size of the book from the glyph heights of its plain paragraphs.

    Usage:
        Call after ``store_glyph_heights`` has run on every page.

    Args:
        page_records: Page records of the whole book with measured segments.

    Returns:
        The statistics, or None when fewer than ``MINIMUM_REFERENCE_SEGMENTS`` paragraphs were measured.
    """
    paragraph_heights = [page_segment[GLYPH_HEIGHT_FIELD] for page_record in page_records
                         for page_segment in page_record["segments"]
                         if page_segment["label"] == BODY_TEXT_LABEL and GLYPH_HEIGHT_FIELD in page_segment]
    if len(paragraph_heights) < MINIMUM_REFERENCE_SEGMENTS:
        return None
    height_quantiles = statistics.quantiles(paragraph_heights, n=FONT_QUANTILE_COUNT)
    return FontStatistics(statistics.median(paragraph_heights), height_quantiles[0], height_quantiles[-1],
                          len(paragraph_heights))


def fits_usual_font(glyph_height: float, font_statistics: FontStatistics) -> bool:
    """Tell whether a glyph height lies in the normal range of the book's paragraphs.

    Args:
        glyph_height: Median glyph height of a segment in pixels.
        font_statistics: Usual font size of the book.

    Returns:
        True if ``lower_height <= glyph_height <= upper_height``.
    """
    return font_statistics.lower_height <= glyph_height <= font_statistics.upper_height


def is_off_size(page_segment: dict, font_statistics: FontStatistics) -> bool:
    """Tell whether a measured body-text segment is set far outside the usual font size.

    Args:
        page_segment: Segment, with ``glyph_height`` when it was measured.
        font_statistics: Usual font size of the book.

    Returns:
        True for body-text segments whose glyph height is outside 0.5x..2x of the typical height.
    """
    glyph_height = page_segment.get(GLYPH_HEIGHT_FIELD)
    if glyph_height is None or position_role(page_segment["label"]) != BODY_TEXT_ROLE:
        return False
    size_ratio = glyph_height / font_statistics.typical_height
    return not OFF_SIZE_LOWER_FACTOR <= size_ratio <= OFF_SIZE_UPPER_FACTOR


def large_lettering_box(page_glyphs: PageGlyphs, font_statistics: FontStatistics) -> list[float] | None:
    """Box around the large letter-like components outside every segment, if there are many of them.

    Args:
        page_glyphs: Measurements of one page.
        font_statistics: Usual font size of the book.

    Returns:
        Union box of the components at least ``LARGE_GLYPH_FACTOR`` x the typical height, or None when there
        are fewer than ``MINIMUM_LARGE_GLYPHS`` of them.
    """
    minimum_height = LARGE_GLYPH_FACTOR * font_statistics.typical_height
    large_glyphs = [glyph_box for glyph_box in page_glyphs.uncovered_glyphs
                    if glyph_box[3] - glyph_box[1] >= minimum_height]
    if len(large_glyphs) < MINIMUM_LARGE_GLYPHS:
        return None
    return union_box([list(glyph_box) for glyph_box in large_glyphs])


def is_inside_frame(region_box: list[float], page_height: float, page_frame: PageFrame | None) -> bool:
    """Tell whether a box stays inside the printed frame of its page group.

    Args:
        region_box: [x1, y1, x2, y2] in page pixels.
        page_height: Page image height in pixels.
        page_frame: Frame of the page's parity group, or None when the group is too small (always inside).

    Returns:
        False if the box protrudes above or below the frame by more than ``FRAME_TOLERANCE_RATIO``.
    """
    if page_frame is None or page_height <= 0:
        return True
    return (page_frame.frame_top - region_box[1] / page_height <= FRAME_TOLERANCE_RATIO
            and region_box[3] / page_height - page_frame.frame_bottom <= FRAME_TOLERANCE_RATIO)


def page_off_size_regions(page_record: dict, page_glyphs: PageGlyphs, font_statistics: FontStatistics,
                          page_frame: PageFrame | None) -> list[OffSizeRegion]:
    """Off-size regions of one page.

    Args:
        page_record: Page record with ``glyph_height`` stored on its measured segments.
        page_glyphs: Measurements of the same page.
        font_statistics: Usual font size of the book.
        page_frame: Printed frame of the page's parity group, or None when unknown.

    Returns:
        One region per off-size segment, plus one for large lettering outside every segment but inside
        the printed frame.
    """
    page_regions = [OffSizeRegion(page_record, list(page_segment["bbox"]), "off-size text",
                                  float(page_segment.get("score") or 0.0))
                    for page_segment in page_record["segments"] if is_off_size(page_segment, font_statistics)]
    lettering_box = large_lettering_box(page_glyphs, font_statistics)
    if lettering_box is not None and is_inside_frame(lettering_box, float(page_record["size"][1]), page_frame):
        page_regions.append(OffSizeRegion(page_record, lettering_box, "large lettering", 0.0))
    return page_regions


def find_off_size_regions(page_records: list[dict], page_measurements: list[PageGlyphs],
                          font_statistics: FontStatistics) -> list[OffSizeRegion]:
    """Find the page regions whose lettering does not fit the usual font size of the book.

    Usage:
        Call after ``store_glyph_heights`` and ``collect_font_statistics``; the pages of the regions are
        segmented again to look for the picture the lettering belongs to.

    Args:
        page_records: Page records of the whole book.
        page_measurements: Measurements aligned with ``page_records``.
        font_statistics: Usual font size of the book.

    Returns:
        Regions in page order.
    """
    frame_by_parity = group_frames(page_records)
    return [off_size_region for page_record, page_glyphs in zip(page_records, page_measurements, strict=True)
            for off_size_region in page_off_size_regions(page_record, page_glyphs, font_statistics,
                                                         frame_by_parity.get(page_record["page"] % 2))]
