"""Map flagged text units to the page: the printed lines to read again, and units with no ink at all.

Units are aligned to the detected word boxes of their segment (``text_styling.token_alignment``, the same
width-based alignment that applies the word styles). A flagged unit's text line (words grouped by their
vertical centres) is re-read; neighbouring flagged lines are merged into one region. A unit has nothing
printed under it when the alignment leaves it without a word box, or when the box it shares with other units
is far narrower than their characters need (the aligner may absorb unprinted text into a neighbour's box).
"""

import statistics
from dataclasses import dataclass

from pdf2fb2.text_styling.token_alignment import TextToken, align_tokens_to_words
from pdf2fb2.text_styling.word_lines import group_into_lines
from pdf2fb2.text_verification.verify_units import VerifyUnit

PADDING_RATIO = 0.35   # of the median word height, around a re-read region
WIDTH_SHORTFALL = 0.5  # a box narrower than this share of the expected width does not hold the text


@dataclass(frozen=True)
class UnitPlacement:
    """Where the units of a segment are on the page.

    Attributes:
        word_lines: Text lines of the segment, words left to right.
        unit_lines: Line index of every unit (None: no word box).
        unit_words: Word index (reading order) of every unit (None: no word box).
        word_widths: Pixel width of every word in reading order.
    """

    word_lines: list[list[dict]]
    unit_lines: list[int | None]
    unit_words: list[int | None]
    word_widths: list[int]


@dataclass(frozen=True)
class ReadRegion:
    """A part of a segment to read again.

    Attributes:
        region_box: ``[x1, y1, x2, y2]`` in page pixels.
        first_unit: First unit of the text the re-read replaces.
        last_unit: Last unit it replaces (inclusive).
    """

    region_box: tuple[int, int, int, int]
    first_unit: int
    last_unit: int


def place_units(verify_units: list[VerifyUnit], segment_words: list[dict]) -> UnitPlacement | None:
    """Align the units with the segment's word boxes.

    Args:
        verify_units: Units of the segment text.
        segment_words: Detected words (``bbox`` in page pixels), any order.

    Returns:
        Placement, or None without words or when no alignment is possible.
    """
    word_lines = group_into_lines(segment_words)
    ordered_words = [(line_index, line_word) for line_index, text_line in enumerate(word_lines)
                     for line_word in text_line]
    word_widths = [line_word["bbox"][2] - line_word["bbox"][0] for _, line_word in ordered_words]
    word_indexes = align_tokens_to_words([TextToken(verify_unit.start_offset, verify_unit.end_offset)
                                          for verify_unit in verify_units], word_widths)
    if word_indexes is None:
        return None
    return UnitPlacement(word_lines, [None if word_index is None else ordered_words[word_index][0]
                                      for word_index in word_indexes], word_indexes, word_widths)


def merged_line_blocks(flagged_lines: set[int]) -> list[list[int]]:
    """Group flagged line indexes into runs of neighbouring lines.

    Args:
        flagged_lines: Line indexes.

    Returns:
        Runs in top-down order.
    """
    line_blocks: list[list[int]] = []
    for line_index in sorted(flagged_lines):
        if line_blocks and line_index == line_blocks[-1][-1] + 1:
            line_blocks[-1].append(line_index)
        else:
            line_blocks.append([line_index])
    return line_blocks


def padded_box(line_words: list[dict], page_size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Bounding box of some words, padded and clamped to the page.

    Args:
        line_words: Words of the region.
        page_size: ``(width, height)`` of the page image.

    Returns:
        Integer box.
    """
    word_heights = sorted(line_word["bbox"][3] - line_word["bbox"][1] for line_word in line_words)
    padding_pixels = round(PADDING_RATIO * word_heights[len(word_heights) // 2])
    return (max(0, min(line_word["bbox"][0] for line_word in line_words) - padding_pixels),
            max(0, min(line_word["bbox"][1] for line_word in line_words) - padding_pixels),
            min(page_size[0], max(line_word["bbox"][2] for line_word in line_words) + padding_pixels),
            min(page_size[1], max(line_word["bbox"][3] for line_word in line_words) + padding_pixels))


def reread_regions(unit_placement: UnitPlacement, flagged_units: list[int],
                   page_size: tuple[int, int]) -> list[ReadRegion]:
    """Regions covering the lines of the flagged units.

    Args:
        unit_placement: Placement of the segment's units.
        flagged_units: Units to re-read.
        page_size: ``(width, height)`` of the page image.

    Returns:
        One region per run of neighbouring flagged lines.
    """
    flagged_lines = {unit_placement.unit_lines[unit_index] for unit_index in flagged_units
                     if unit_placement.unit_lines[unit_index] is not None}
    read_regions = []
    for line_block in merged_line_blocks(flagged_lines):
        block_units = [unit_index for unit_index, line_index in enumerate(unit_placement.unit_lines)
                       if line_index in line_block]
        block_words = [line_word for line_index in line_block for line_word in unit_placement.word_lines[line_index]]
        read_regions.append(ReadRegion(padded_box(block_words, page_size), min(block_units), max(block_units)))
    return read_regions


def pixels_per_character(unit_placement: UnitPlacement, verify_units: list[VerifyUnit],
                         suspect_units: set[int]) -> float | None:
    """Median width per character of trusted units that alone fill their word box.

    Args:
        unit_placement: Placement of the segment's units.
        verify_units: Units of the segment.
        suspect_units: Units left out of the estimate (flagged as not on the page).

    Returns:
        Pixels per character, or None without any trusted one-to-one unit.
    """
    word_users = [unit_placement.unit_words.count(word_index) for word_index in range(len(unit_placement.word_widths))]
    single_rates = [unit_placement.word_widths[word_index] / len(verify_unit.unit_text)
                    for unit_index, (verify_unit, word_index) in enumerate(zip(verify_units, unit_placement.unit_words,
                                                                               strict=True))
                    if word_index is not None and word_users[word_index] == 1 and unit_index not in suspect_units]
    return statistics.median(single_rates) if single_rates else None


def unprinted_units(unit_placement: UnitPlacement | None, inserted_units: list[int],
                    verify_units: list[VerifyUnit]) -> list[int]:
    """Inserted units that have no printed text under them.

    Args:
        unit_placement: Placement (None: unknown, nothing is confirmed).
        inserted_units: Units the verifier flags as not on the page.
        verify_units: Units of the segment.

    Returns:
        Units both signals agree on: no box, or a box far too narrow for the units mapped onto it.
    """
    if unit_placement is None:
        return []
    character_width = pixels_per_character(unit_placement, verify_units, set(inserted_units))
    confirmed_units = []
    for unit_index in inserted_units:
        word_index = unit_placement.unit_words[unit_index]
        if word_index is None:
            confirmed_units.append(unit_index)
            continue
        sharing_units = [verify_units[shared_index].unit_text for shared_index, shared_word
                         in enumerate(unit_placement.unit_words) if shared_word == word_index]
        expected_width = (character_width or 0.0) * len(" ".join(sharing_units))
        if unit_placement.word_widths[word_index] < WIDTH_SHORTFALL * expected_width:
            confirmed_units.append(unit_index)
    return confirmed_units
