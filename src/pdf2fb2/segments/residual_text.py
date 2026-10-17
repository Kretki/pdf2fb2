"""Residual text: letters of the usual font size left outside every segment once a page is fully processed.

With all segments blanked, the letter-like ink of the usual size (0.5x..2x the usual glyph height) is what the
layout passes missed. It is grouped into lines (glyphs on one baseline, split at gaps wider than 3 glyph
heights) and lines into blocks (line centres at most 3.2 glyph heights apart, overlapping horizontally):

* line pieces in the band where the book prints its running heads (at least 3 glyphs) or page numbers (at most
  8 glyphs, at the usual horizontal position) are those elements, missed by the layout model; the bands are the
  median positions of the ``header`` / ``number`` segments of the page group;
* blocks of at least 12 glyphs in lines of at least 6, inside the printed frame, are missed paragraphs.
"""

import statistics
from collections import defaultdict
from dataclasses import dataclass

from pdf2fb2.segments.font_statistics import OFF_SIZE_LOWER_FACTOR, OFF_SIZE_UPPER_FACTOR, FontStatistics
from pdf2fb2.segments.label_roles import PAGE_NUMBER_LABELS
from pdf2fb2.segments.position_statistics import MINIMUM_GROUP_SIZE, center_ratio
from pdf2fb2.segments.segment_geometry import union_box

GlyphBox = tuple[int, int, int, int]
GlyphLine = list[GlyphBox]
HEADER_LABEL = "header"
NUMBER_LABEL = "number"
TEXT_LABEL = "text"
LINE_CENTER_FACTOR = 0.6  # glyph centre distance to the line centre, in usual glyph heights
WORD_GAP_FACTOR = 5.0  # horizontal gap that splits a line, in usual glyph heights
LINE_PITCH_FACTOR = 3.2  # largest centre distance of consecutive lines of one block, in usual glyph heights
MINIMUM_LINE_GLYPHS = 6
MINIMUM_BLOCK_GLYPHS = 12
MAXIMUM_NUMBER_GLYPHS = 8
MINIMUM_HEADER_GLYPHS = 3
NUMBER_CENTER_TOLERANCE = 0.08  # fraction of the page width around the usual horizontal page-number position
BAND_TOLERANCE_RATIO = 0.01  # fraction of the page height around the median band of the page group


@dataclass(frozen=True)
class ResidualBlock:
    """A group of residual glyphs that should be a segment.

    Attributes:
        block_box: [x1, y1, x2, y2] of the glyphs in page pixels.
        block_label: ``text`` for a missed paragraph, ``header`` / ``number`` for a missed running head or
            page number.
        glyph_count: Number of glyphs in the block.
        line_count: Number of text lines in the block.
    """

    block_box: list[float]
    block_label: str
    glyph_count: int
    line_count: int


@dataclass(frozen=True)
class MarginBands:
    """Vertical bands of the running heads and page numbers of one page group, as fractions of the page height.

    Attributes:
        header_band: (top, bottom) of the running-head band, or None when the group has too few running heads.
        number_band: (top, bottom) of the page-number band, or None when the group has too few page numbers.
        number_center: Median horizontal centre of the page numbers as a fraction of the page width, or None.
    """

    header_band: tuple[float, float] | None
    number_band: tuple[float, float] | None
    number_center: float | None


def usual_glyphs(glyph_boxes: list[GlyphBox], font_statistics: FontStatistics) -> list[GlyphBox]:
    """Glyphs whose height is within 0.5x..2x of the usual glyph height.

    Args:
        glyph_boxes: [x1, y1, x2, y2] glyph boxes.
        font_statistics: Usual font size of the book.

    Returns:
        The glyphs of a regular text size.
    """
    return [glyph_box for glyph_box in glyph_boxes
            if OFF_SIZE_LOWER_FACTOR * font_statistics.typical_height <= glyph_box[3] - glyph_box[1]
            <= OFF_SIZE_UPPER_FACTOR * font_statistics.typical_height]


def median_band(band_extents: list[tuple[float, float]]) -> tuple[float, float] | None:
    """Median band of a list of vertical extents, widened by the tolerance.

    Args:
        band_extents: (top, bottom) fractions of the page height.

    Returns:
        (top, bottom) of the band, or None for fewer than ``MINIMUM_GROUP_SIZE`` extents.
    """
    if len(band_extents) < MINIMUM_GROUP_SIZE:
        return None
    return (statistics.median(extent_pair[0] for extent_pair in band_extents) - BAND_TOLERANCE_RATIO,
            statistics.median(extent_pair[1] for extent_pair in band_extents) + BAND_TOLERANCE_RATIO)


def collect_margin_bands(page_records: list[dict]) -> dict[int, MarginBands]:
    """Running-head and page-number bands of every page group (page parity).

    Args:
        page_records: Segmented page records of the book.

    Returns:
        Mapping page parity (0/1) -> bands.
    """
    extents_by_key: dict[tuple[int, str], list[tuple[float, float]]] = defaultdict(list)
    number_centers: dict[int, list[float]] = defaultdict(list)
    for page_record in page_records:
        page_width, page_height = float(page_record["size"][0]), float(page_record["size"][1])
        for page_segment in page_record["segments"]:
            segment_band = HEADER_LABEL if page_segment["label"] == HEADER_LABEL else (
                NUMBER_LABEL if page_segment["label"] in PAGE_NUMBER_LABELS else None)
            if segment_band is None or page_height <= 0 or page_width <= 0:
                continue
            extents_by_key[(page_record["page"] % 2, segment_band)].append(
                (page_segment["bbox"][1] / page_height, page_segment["bbox"][3] / page_height))
            if segment_band == NUMBER_LABEL:
                number_centers[page_record["page"] % 2].append(center_ratio(page_segment["bbox"], page_width))
    return {page_parity: MarginBands(median_band(extents_by_key[(page_parity, HEADER_LABEL)]),
                                     median_band(extents_by_key[(page_parity, NUMBER_LABEL)]),
                                     statistics.median(number_centers[page_parity])
                                     if number_centers[page_parity] else None)
            for page_parity in (0, 1)}


def glyph_center(glyph_box: GlyphBox) -> float:
    """Vertical centre of a glyph.

    Args:
        glyph_box: [x1, y1, x2, y2].

    Returns:
        (y1 + y2) / 2 in pixels.
    """
    return (glyph_box[1] + glyph_box[3]) / 2


def split_at_gaps(line_glyphs: GlyphLine, maximum_gap: float) -> list[GlyphLine]:
    """Split one line into pieces at wide horizontal gaps.

    Args:
        line_glyphs: Glyphs of one line.
        maximum_gap: Largest gap inside a piece, in pixels.

    Returns:
        Pieces in left-to-right order.
    """
    line_pieces: list[GlyphLine] = []
    for glyph_box in sorted(line_glyphs):
        if line_pieces and glyph_box[0] - max(piece_glyph[2] for piece_glyph in line_pieces[-1]) <= maximum_gap:
            line_pieces[-1].append(glyph_box)
        else:
            line_pieces.append([glyph_box])
    return line_pieces


def group_lines(glyph_boxes: list[GlyphBox], font_statistics: FontStatistics) -> list[GlyphLine]:
    """Group glyphs into line pieces.

    Args:
        glyph_boxes: Residual glyphs.
        font_statistics: Usual font size of the book.

    Returns:
        Line pieces, top to bottom.
    """
    typical_height = font_statistics.typical_height
    glyph_lines: list[GlyphLine] = []
    for glyph_box in sorted(glyph_boxes, key=glyph_center):
        if glyph_lines and abs(glyph_center(glyph_box) - glyph_center(glyph_lines[-1][-1])) <= (
                LINE_CENTER_FACTOR * typical_height):
            glyph_lines[-1].append(glyph_box)
        else:
            glyph_lines.append([glyph_box])
    return [line_piece for glyph_line in glyph_lines
            for line_piece in split_at_gaps(glyph_line, WORD_GAP_FACTOR * typical_height)]


def piece_box(line_piece: GlyphLine) -> list[float]:
    """Box of a line piece.

    Args:
        line_piece: Glyphs of the piece.

    Returns:
        [x1, y1, x2, y2] enclosing them.
    """
    return union_box([list(glyph_box) for glyph_box in line_piece])


def lies_in_band(line_box: list[float], page_height: float, vertical_band: tuple[float, float] | None) -> bool:
    """Tell whether the vertical centre of a box lies in a band.

    Args:
        line_box: [x1, y1, x2, y2] in page pixels.
        page_height: Page image height in pixels (> 0).
        vertical_band: (top, bottom) as fractions of the page height, or None.

    Returns:
        True if the band is known and contains the box centre.
    """
    return vertical_band is not None and vertical_band[0] <= (line_box[1] + line_box[3]) / 2 / page_height \
        <= vertical_band[1]


def is_number_piece(line_piece: GlyphLine, page_size: tuple[float, float], margin_bands: MarginBands) -> bool:
    """Tell whether a line piece is a missed page number.

    Args:
        line_piece: Glyphs of one line piece.
        page_size: (width, height) of the page in pixels (both > 0).
        margin_bands: Bands of the page group.

    Returns:
        True for pieces of at most ``MAXIMUM_NUMBER_GLYPHS`` glyphs in the number band, at the usual horizontal
        position of the page numbers.
    """
    line_box = piece_box(line_piece)
    return (len(line_piece) <= MAXIMUM_NUMBER_GLYPHS and margin_bands.number_center is not None
            and lies_in_band(line_box, page_size[1], margin_bands.number_band)
            and abs(center_ratio(line_box, page_size[0]) - margin_bands.number_center) <= NUMBER_CENTER_TOLERANCE)


def band_label(line_piece: GlyphLine, page_size: tuple[float, float], margin_bands: MarginBands | None) -> str | None:
    """Tell whether a line piece is a missed running head or page number.

    Args:
        line_piece: Glyphs of one line piece.
        page_size: (width, height) of the page in pixels.
        margin_bands: Bands of the page group, or None.

    Returns:
        ``header`` for pieces of at least ``MINIMUM_HEADER_GLYPHS`` glyphs in the running-head band, ``number``
        for page-number pieces, else None.
    """
    if margin_bands is None or page_size[0] <= 0 or page_size[1] <= 0:
        return None
    if len(line_piece) >= MINIMUM_HEADER_GLYPHS and lies_in_band(piece_box(line_piece), page_size[1],
                                                            margin_bands.header_band):
        return HEADER_LABEL
    return NUMBER_LABEL if is_number_piece(line_piece, page_size, margin_bands) else None


def overlaps_horizontally(first_box: list[float], second_box: list[float]) -> bool:
    """Tell whether two boxes share a horizontal range.

    Args:
        first_box: [x1, y1, x2, y2].
        second_box: [x1, y1, x2, y2].

    Returns:
        True if their x ranges intersect.
    """
    return min(first_box[2], second_box[2]) > max(first_box[0], second_box[0])


def group_blocks(text_lines: list[GlyphLine], font_statistics: FontStatistics) -> list[list[GlyphLine]]:
    """Group line pieces into blocks of consecutive, horizontally overlapping lines.

    Args:
        text_lines: Line pieces with enough glyphs, top to bottom.
        font_statistics: Usual font size of the book.

    Returns:
        Blocks, each a list of line pieces.
    """
    maximum_pitch = LINE_PITCH_FACTOR * font_statistics.typical_height
    line_blocks: list[list[GlyphLine]] = []
    for line_piece in text_lines:
        current_box = piece_box(line_piece)
        joined_block = next((line_block for line_block in reversed(line_blocks)
                             if 0 <= (current_box[1] + current_box[3]) / 2 - (
                                 piece_box(line_block[-1])[1] + piece_box(line_block[-1])[3]) / 2 <= maximum_pitch
                             and overlaps_horizontally(current_box, union_box([piece_box(block_line)
                                                                              for block_line in line_block]))), None)
        if joined_block is None:
            line_blocks.append([line_piece])
        else:
            joined_block.append(line_piece)
    return line_blocks


def text_blocks(text_lines: list[GlyphLine], font_statistics: FontStatistics) -> list[ResidualBlock]:
    """Missed paragraphs among the line pieces.

    Args:
        text_lines: Line pieces outside the margin bands, top to bottom.
        font_statistics: Usual font size of the book.

    Returns:
        Blocks with at least ``MINIMUM_BLOCK_GLYPHS`` glyphs in lines of at least ``MINIMUM_LINE_GLYPHS``.
    """
    full_lines = [line_piece for line_piece in text_lines if len(line_piece) >= MINIMUM_LINE_GLYPHS]
    residual_blocks = []
    for line_block in group_blocks(full_lines, font_statistics):
        glyph_count = sum(len(line_piece) for line_piece in line_block)
        if glyph_count >= MINIMUM_BLOCK_GLYPHS:
            residual_blocks.append(ResidualBlock(union_box([piece_box(line_piece) for line_piece in line_block]),
                                                 TEXT_LABEL, glyph_count, len(line_block)))
    return residual_blocks


def find_residual_blocks(page_record: dict, glyph_boxes: list[GlyphBox], font_statistics: FontStatistics,
                         margin_bands: MarginBands | None) -> list[ResidualBlock]:
    """Missed running heads, page numbers and paragraphs of one page.

    Usage:
        Call with the glyphs of the usual size left outside every segment.

    Args:
        page_record: Page record.
        glyph_boxes: Residual glyphs of the usual size.
        font_statistics: Usual font size of the book.
        margin_bands: Bands of the page group, or None.

    Returns:
        Blocks to add as segments, band elements first.
    """
    page_size = (float(page_record["size"][0]), float(page_record["size"][1]))
    band_blocks = []
    remaining_lines = []
    for line_piece in group_lines(glyph_boxes, font_statistics):
        piece_label = band_label(line_piece, page_size, margin_bands)
        if piece_label is None:
            remaining_lines.append(line_piece)
        else:
            band_blocks.append(ResidualBlock(piece_box(line_piece), piece_label, len(line_piece), 1))
    return band_blocks + text_blocks(remaining_lines, font_statistics)
