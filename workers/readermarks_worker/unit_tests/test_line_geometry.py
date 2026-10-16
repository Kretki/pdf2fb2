"""Unit tests of the geometric hand-line vs printed-rule filter."""

import numpy

from readermarks.postprocess.line_geometry import (
    HAND_LINE_VERDICT,
    NOT_LINE_VERDICT,
    PRINTED_RULE_VERDICT,
    classify_line,
    measure_line,
)
from readermarks.settings.model_settings import CleaningSettings


def straight_rule_mask(rule_length: int, rule_thickness: int) -> numpy.ndarray:
    """Perfectly straight, uniform horizontal rule.

    Args:
        rule_length: Length in pixels.
        rule_thickness: Thickness in pixels.

    Returns:
        (thickness, length) bool mask.
    """
    return numpy.ones((rule_thickness, rule_length), bool)


def wobbly_line_mask(line_length: int, random_seed: int) -> numpy.ndarray:
    """Hand-drawn-looking line: sloped, wavy, varying thickness.

    Args:
        line_length: Length in pixels.
        random_seed: Seed.

    Returns:
        Bool mask.
    """
    random_source = numpy.random.default_rng(random_seed)
    draw_canvas = numpy.zeros((30, line_length), bool)
    for column_index in range(line_length):
        center_row = 12 + 0.06 * column_index + 2.5 * numpy.sin(column_index / 9.0)
        half_width = 1 + int(random_source.integers(0, 3))
        draw_canvas[int(center_row) - half_width : int(center_row) + half_width + 1, column_index] = True
    filled_rows, filled_columns = numpy.nonzero(draw_canvas)
    return draw_canvas[filled_rows.min() : filled_rows.max() + 1, filled_columns.min() : filled_columns.max() + 1]


def test_straight_rule_is_printed_rule() -> None:
    """A long, flat, uniform line is classified as a printed rule."""
    rule_mask = straight_rule_mask(200, 2)
    line_verdict = classify_line(rule_mask, (100, 50, 300, 52), [], CleaningSettings())
    assert line_verdict == PRINTED_RULE_VERDICT


def test_wobbly_line_is_hand_line() -> None:
    """A sloped, wavy line of varying thickness is classified as hand-drawn."""
    line_mask = wobbly_line_mask(220, 7)
    line_verdict = classify_line(
        line_mask, (10, 10, 10 + line_mask.shape[1], 10 + line_mask.shape[0]), [], CleaningSettings()
    )
    assert line_verdict == HAND_LINE_VERDICT


def test_blob_is_not_a_line() -> None:
    """A compact blob (e.g. a handwritten letter) is not measured as a line."""
    blob_mask = numpy.ones((20, 25), bool)
    assert measure_line(blob_mask) is None
    assert classify_line(blob_mask, (0, 0, 25, 20), [], CleaningSettings()) == NOT_LINE_VERDICT


def test_straight_line_overshooting_words_is_hand_line() -> None:
    """A straight line that runs far past the underlined words counts as hand-drawn."""
    rule_mask = straight_rule_mask(200, 2)
    word_boxes = [(140, 30, 200, 48), (205, 30, 250, 48)]
    line_verdict = classify_line(rule_mask, (100, 50, 300, 52), word_boxes, CleaningSettings())
    assert line_verdict == HAND_LINE_VERDICT


def test_exact_width_underline_is_printed_rule() -> None:
    """A straight underline spanning exactly its words stays a printed rule."""
    rule_mask = straight_rule_mask(150, 1)
    word_boxes = [(100, 30, 170, 48), (176, 30, 250, 48)]
    line_verdict = classify_line(rule_mask, (100, 50, 250, 51), word_boxes, CleaningSettings())
    assert line_verdict == PRINTED_RULE_VERDICT
