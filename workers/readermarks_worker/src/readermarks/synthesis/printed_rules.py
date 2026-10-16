"""Synthetic printed rules: exact-width typographic underlines, footnote separators, table rules.

Printed rules are perfectly straight, uniform in thickness, printed in the page's ink color and
aligned with the text block. They are hard negatives (``printed_rule`` + ``printed_ink``).
"""

import numpy

from readermarks.synthesis.mark_placement import random_span
from readermarks.synthesis.page_analysis import PageAnalysis, PixelBox


def draw_rule(
    page_image: numpy.ndarray,
    ink_mask: numpy.ndarray,
    rule_mask: numpy.ndarray,
    rule_box: PixelBox,
    ink_color: tuple[int, int, int],
) -> None:
    """Paint one axis-aligned rule rectangle and label it.

    Args:
        page_image: (H, W, 3) float32 page 0..1, mutated.
        ink_mask: (H, W) uint8 printed-ink mask, mutated.
        rule_mask: (H, W) uint8 printed-rule mask, mutated.
        rule_box: Rectangle (x_start, y_start, x_end, y_end), end exclusive.
        ink_color: RGB ink color, 0..255.

    Returns:
        None.
    """
    page_height, page_width = ink_mask.shape
    left_edge, top_edge = max(0, rule_box[0]), max(0, rule_box[1])
    right_edge, bottom_edge = min(page_width, rule_box[2]), min(page_height, rule_box[3])
    if right_edge <= left_edge or bottom_edge <= top_edge:
        return
    page_image[top_edge:bottom_edge, left_edge:right_edge] = numpy.minimum(
        page_image[top_edge:bottom_edge, left_edge:right_edge], numpy.array(ink_color, numpy.float32) / 255.0
    )
    ink_mask[top_edge:bottom_edge, left_edge:right_edge] = 255
    rule_mask[top_edge:bottom_edge, left_edge:right_edge] = 255


def typographic_underline_boxes(analysis: PageAnalysis, random_source: numpy.random.Generator) -> list[PixelBox]:
    """Exact-width underlines just below 1..4 printed spans.

    Args:
        analysis: Page analysis.
        random_source: Random generator.

    Returns:
        Rule rectangles.
    """
    rule_boxes = []
    for _ in range(int(random_source.integers(1, 5))):
        span_box = random_span(analysis, random_source, 5)
        if span_box is not None:
            thickness = int(random_source.integers(1, 3))
            baseline = span_box[3] + int(random_source.integers(0, 3))
            rule_boxes.append((span_box[0], baseline, span_box[2], baseline + thickness))
    return rule_boxes


def separator_boxes(analysis: PageAnalysis, random_source: numpy.random.Generator) -> list[PixelBox]:
    """A footnote separator or ornament rule in the text block.

    Args:
        analysis: Page analysis.
        random_source: Random generator.

    Returns:
        Rule rectangles (one, or two for a double ornament rule).
    """
    text_box = analysis.text_box
    text_width = text_box[2] - text_box[0]
    if not analysis.line_boxes:
        return []
    anchor_line = analysis.line_boxes[int(random_source.integers(len(analysis.line_boxes)))]
    rule_top = anchor_line[1] - int(random_source.integers(3, 8))
    thickness = int(random_source.integers(1, 3))
    if random_source.random() < 0.5:
        rule_length = int(text_width * random_source.uniform(0.15, 0.35))
        return [(text_box[0], rule_top, text_box[0] + rule_length, rule_top + thickness)]
    rule_length = int(text_width * random_source.uniform(0.1, 0.3))
    center_x = (text_box[0] + text_box[2]) // 2
    first_rule = (center_x - rule_length // 2, rule_top, center_x + rule_length // 2, rule_top + thickness)
    return [first_rule, (first_rule[0], rule_top + thickness + 2, first_rule[2], rule_top + 2 * thickness + 2)]


def table_rule_boxes(analysis: PageAnalysis, random_source: numpy.random.Generator) -> list[PixelBox]:
    """Full-width horizontal table rules (and sometimes column rules) around a run of lines.

    Args:
        analysis: Page analysis.
        random_source: Random generator.

    Returns:
        Rule rectangles.
    """
    if len(analysis.line_boxes) < 3:
        return []
    text_box = analysis.text_box
    first_line = int(random_source.integers(0, len(analysis.line_boxes) - 2))
    last_line = min(len(analysis.line_boxes) - 1, first_line + int(random_source.integers(2, 8)))
    upper_row_y = analysis.line_boxes[first_line][1] - 3
    bottom_y = analysis.line_boxes[last_line][3] + 3
    thickness = int(random_source.integers(1, 3))
    rule_boxes = [
        (text_box[0], upper_row_y, text_box[2], upper_row_y + thickness),
        (text_box[0], bottom_y, text_box[2], bottom_y + thickness),
    ]
    if random_source.random() < 0.4:
        column_x = int(random_source.uniform(text_box[0] + 40, text_box[2] - 40))
        rule_boxes.append((column_x, upper_row_y, column_x + 1, bottom_y))
    return rule_boxes


RULE_BUILDERS = (typographic_underline_boxes, separator_boxes, table_rule_boxes)


def add_printed_rules(
    page_image: numpy.ndarray,
    ink_mask: numpy.ndarray,
    rule_mask: numpy.ndarray,
    analysis: PageAnalysis,
    random_source: numpy.random.Generator,
) -> None:
    """Add 1..3 kinds of printed rules to a page (hard negatives).

    Usage:
        Called by the sample generator for pages chosen as printed-rule pages.

    Args:
        page_image: (H, W, 3) float32 page 0..1, mutated.
        ink_mask: (H, W) uint8 printed-ink mask, mutated.
        rule_mask: (H, W) uint8 printed-rule mask, mutated.
        analysis: Page analysis.
        random_source: Random generator.

    Returns:
        None.
    """
    builder_count = int(random_source.integers(1, len(RULE_BUILDERS) + 1))
    for builder_index in random_source.choice(len(RULE_BUILDERS), builder_count, replace=False):
        for rule_box in RULE_BUILDERS[builder_index](analysis, random_source):
            draw_rule(page_image, ink_mask, rule_mask, rule_box, analysis.ink_color)
