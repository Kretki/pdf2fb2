"""Page evidence of every verifier unit: OCR confidence, word boxes, second read, page context and word crops.

The same code builds the evidence for pipeline jobs (``feedback_loop``) and for training data
(``benchmark.verifier_export``), so the verifier sees identical features in training and in use. Every feature is
language-independent (``worker_protocol.unit_features``):

* OCR confidence: the raw model output's per-character log-probabilities (``segment["ocr_confidence"]``) are
  mapped to the final text by character alignment (cleaning shifts offsets), then reduced per unit;
* word boxes: from the unit placement (``region_mapping.place_units``): no box under the unit, units per box,
  box width against the width the characters need;
* second read: whether a second OCR read of the segment (``segment["second_read_text"]``) has the same unit;
* context: units of the neighbouring book-text segments of the same page;
* crops: one image per detected word, written under ``verify_crops/`` (opt-in).
"""

import difflib
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from pdf2fb2.text_verification.region_mapping import UnitPlacement, pixels_per_character
from pdf2fb2.text_verification.verify_units import VerifyUnit, verify_units

CONTEXT_UNITS = 32
CROP_DIRECTORY = "verify_crops"
CROP_PADDING_RATIO = 0.25
MAX_CROP_SIZE = (384, 96)
MAX_ALIGNED_CHARACTERS = 50_000


def character_map(final_text: str, raw_text: str) -> dict[int, int]:
    """Positions of the final text's characters in the raw OCR output.

    Args:
        final_text: Segment text after cleaning.
        raw_text: Model output the confidences belong to.

    Returns:
        Final index -> raw index for characters in matching blocks.
    """
    if len(final_text) > MAX_ALIGNED_CHARACTERS or len(raw_text) > MAX_ALIGNED_CHARACTERS:
        return {}
    text_matcher = difflib.SequenceMatcher(None, final_text, raw_text, autojunk=False)
    return {final_start + char_offset: raw_start + char_offset
            for final_start, raw_start, block_size in text_matcher.get_matching_blocks()
            for char_offset in range(block_size)}


def confidence_columns(segment_text: str, segment_units: Sequence[VerifyUnit],
                       ocr_confidence: dict | None) -> dict[str, tuple[float | None, ...]]:
    """Lowest and mean OCR log-probability of every unit.

    Args:
        segment_text: Text the units were cut from.
        segment_units: Units with character offsets.
        ocr_confidence: ``{"raw_text", "char_logprobs"}`` of the segment, or None.

    Returns:
        ``{"ocr_min_logprob": ..., "ocr_mean_logprob": ...}`` (empty without confidences).
    """
    if not ocr_confidence:
        return {}
    char_logprobs = ocr_confidence["char_logprobs"]
    position_map = character_map(segment_text, ocr_confidence["raw_text"])
    lowest_values, mean_values = [], []
    for verify_unit in segment_units:
        unit_values = [float(char_logprobs[position_map[char_index]])
                       for char_index in range(verify_unit.start_offset, verify_unit.end_offset)
                       if char_index in position_map]
        lowest_values.append(min(unit_values) if unit_values else None)
        mean_values.append(sum(unit_values) / len(unit_values) if unit_values else None)
    return {"ocr_min_logprob": tuple(lowest_values), "ocr_mean_logprob": tuple(mean_values)}


def width_ratios(unit_placement: UnitPlacement, segment_units: Sequence[VerifyUnit]) -> tuple[float | None, ...]:
    """Box width of every unit divided by the width its box's units need at this page's character width.

    Args:
        unit_placement: Placement of the segment's units.
        segment_units: Units of the segment.

    Returns:
        Ratio per unit (None without a box or without a character-width estimate).
    """
    character_width = pixels_per_character(unit_placement, list(segment_units), set())
    box_texts: dict[int, list[str]] = {}
    for verify_unit, word_index in zip(segment_units, unit_placement.unit_words, strict=True):
        if word_index is not None:
            box_texts.setdefault(word_index, []).append(verify_unit.unit_text)
    return tuple(None if word_index is None or not character_width else
                 unit_placement.word_widths[word_index] / (character_width * len(" ".join(box_texts[word_index])))
                 for word_index in unit_placement.unit_words)


def box_columns(unit_placement: UnitPlacement | None,
                segment_units: Sequence[VerifyUnit]) -> dict[str, tuple[float | None, ...]]:
    """Word-box evidence of every unit.

    Args:
        unit_placement: Placement of the segment's units, or None (no words / no alignment).
        segment_units: Units of the segment.

    Returns:
        ``{"box_missing", "box_shared_units", "box_width_ratio"}`` columns (empty without a placement).
    """
    if unit_placement is None or len(unit_placement.unit_words) != len(segment_units):
        return {}
    word_usage = Counter(word_index for word_index in unit_placement.unit_words if word_index is not None)
    return {"box_missing": tuple(1.0 if word_index is None else 0.0 for word_index in unit_placement.unit_words),
            "box_shared_units": tuple(None if word_index is None else float(word_usage[word_index])
                                      for word_index in unit_placement.unit_words),
            "box_width_ratio": width_ratios(unit_placement, segment_units)}


def reread_columns(segment_units: Sequence[VerifyUnit],
                   second_read_text: str | None) -> dict[str, tuple[float | None, ...]]:
    """Whether a second OCR read disagrees with every unit.

    Args:
        segment_units: Units of the first read.
        second_read_text: Text of the second read, or None.

    Returns:
        ``{"reread_disagrees": ...}`` (empty without a second read).
    """
    if second_read_text is None:
        return {}
    first_units = [verify_unit.unit_text for verify_unit in segment_units]
    second_units = [verify_unit.unit_text for verify_unit in verify_units(second_read_text)]
    unit_matcher = difflib.SequenceMatcher(None, first_units, second_units, autojunk=False)
    agreeing_units = {first_start + unit_offset for first_start, _, block_size in unit_matcher.get_matching_blocks()
                      for unit_offset in range(block_size)}
    return {"reread_disagrees": tuple(0.0 if unit_index in agreeing_units else 1.0
                                      for unit_index in range(len(first_units)))}


def unit_features(page_segment: dict, segment_text: str, segment_units: Sequence[VerifyUnit],
                  unit_placement: UnitPlacement | None) -> dict[str, tuple[float | None, ...]]:
    """All page-evidence columns of a segment.

    Usage:
        Called per segment by the feedback loop and by the verifier export.

    Args:
        page_segment: Segment record (optional ``ocr_confidence`` and ``second_read_text``).
        segment_text: Text the units were cut from (the unstyled OCR text).
        segment_units: Units of ``segment_text``.
        unit_placement: Placement on the word boxes, or None.

    Returns:
        Feature columns (only those with evidence).
    """
    return {**confidence_columns(segment_text, segment_units, page_segment.get("ocr_confidence")),
            **box_columns(unit_placement, segment_units),
            **reread_columns(segment_units, page_segment.get("second_read_text"))}


def page_contexts(page_units: list[list[str]]) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """Unlabelled context of every segment of one page: units of the segments before and after it.

    Args:
        page_units: Units of the page's verified segments in reading order.

    Returns:
        ``(before, after)`` per segment, at most ``CONTEXT_UNITS`` units each.
    """
    flat_units = [unit_text for segment_list in page_units for unit_text in segment_list]
    segment_contexts, read_offset = [], 0
    for segment_list in page_units:
        segment_end = read_offset + len(segment_list)
        segment_contexts.append((tuple(flat_units[max(0, read_offset - CONTEXT_UNITS):read_offset]),
                                 tuple(flat_units[segment_end:segment_end + CONTEXT_UNITS])))
        read_offset = segment_end
    return segment_contexts


def padded_word_box(word_box: Sequence[float], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Word box with a margin, clipped to the image.

    Args:
        word_box: ``[x1, y1, x2, y2]`` in page pixels.
        image_size: ``(width, height)`` of the page image.

    Returns:
        Integer crop box.
    """
    margin_size = int((word_box[3] - word_box[1]) * CROP_PADDING_RATIO) + 1
    return (max(0, int(word_box[0]) - margin_size), max(0, int(word_box[1]) - margin_size),
            min(image_size[0], int(word_box[2]) + margin_size), min(image_size[1], int(word_box[3]) + margin_size))


def unit_crops(page_image: Image.Image, unit_placement: UnitPlacement | None, crop_stem: str,
               output_root: Path) -> tuple[str | None, ...]:
    """Write one crop per placed word and return each unit's crop path.

    Args:
        page_image: Page image the boxes refer to.
        unit_placement: Placement of the segment's units, or None.
        crop_stem: File-name stem unique to the segment (safe characters only).
        output_root: Directory the returned paths are relative to (crops go to ``verify_crops/`` inside it).

    Returns:
        Relative crop path or None per unit (empty without a placement).
    """
    if unit_placement is None:
        return ()
    ordered_words = [line_word for text_line in unit_placement.word_lines for line_word in text_line]
    crop_directory = output_root / CROP_DIRECTORY
    crop_directory.mkdir(parents=True, exist_ok=True)
    written_paths: dict[int, str] = {}
    for word_index in sorted({word_index for word_index in unit_placement.unit_words if word_index is not None}):
        crop_image = page_image.crop(padded_word_box(ordered_words[word_index]["bbox"], page_image.size))
        crop_image.thumbnail(MAX_CROP_SIZE)
        relative_path = f"{CROP_DIRECTORY}/{crop_stem}_{word_index:04d}.png"
        crop_image.convert("L").save(output_root / relative_path)
        written_paths[word_index] = relative_path
    return tuple(None if word_index is None else written_paths[word_index]
                 for word_index in unit_placement.unit_words)
