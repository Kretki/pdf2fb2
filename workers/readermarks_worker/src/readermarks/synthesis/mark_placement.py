"""Mark planners: choose where each mark class goes relative to the page's lines and words.

Each planner renders the coverage (alpha) of one mark of its class; handwriting planners live in
``handwriting_placement`` and all planners are registered in ``mark_registry``.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy

from readermarks.synthesis.ink_compositing import DrawingTool
from readermarks.synthesis.page_analysis import PageAnalysis, PixelBox, union_box
from readermarks.synthesis.stroke_geometry import (
    bracket_paths,
    underline_paths,
)
from readermarks.synthesis.stroke_rendering import LocalAlpha, render_polylines
from readermarks.synthesis.symbol_geometry import symbol_paths

MarkPlanner = Callable[[PageAnalysis, tuple[int, int], DrawingTool, numpy.random.Generator], list[LocalAlpha]]


@dataclass(frozen=True)
class PlannerEntry:
    """A registered mark planner.

    Attributes:
        class_name: Mark class it produces (from ``MARK_CLASSES``).
        plan_function: Function that renders the coverage of one or more marks.
        sampling_weight: Relative frequency of the class on synthetic pages.
        tool_kind: Forced tool kind (e.g. highlighter), or None for the page's pencil/pen.
    """

    class_name: str
    plan_function: MarkPlanner
    sampling_weight: float
    tool_kind: str | None = None


def words_of_line(analysis: PageAnalysis, line_box: PixelBox) -> list[PixelBox]:
    """Word boxes whose vertical center lies inside a line box, left to right.

    Args:
        analysis: Page analysis.
        line_box: One of ``analysis.line_boxes``.

    Returns:
        Word boxes of the line (may be empty).
    """
    line_words = [
        word_box
        for word_box in analysis.word_boxes
        if line_box[1] <= (word_box[1] + word_box[3]) / 2 <= line_box[3]
        and word_box[0] >= line_box[0] - 2
        and word_box[2] <= line_box[2] + 2
    ]
    return sorted(line_words)


def random_span(
    analysis: PageAnalysis, random_source: numpy.random.Generator, maximum_words: int, line_index: int | None = None
) -> PixelBox | None:
    """Pick consecutive words on a random (or given) line.

    Args:
        analysis: Page analysis.
        random_source: Random generator.
        maximum_words: Maximum number of words in the span.
        line_index: Line to use; None picks one at random.

    Returns:
        Box of the span, or None when the page has no usable line.
    """
    if not analysis.line_boxes:
        return None
    chosen_index = int(random_source.integers(len(analysis.line_boxes))) if line_index is None else line_index
    line_box = analysis.line_boxes[chosen_index]
    line_words = words_of_line(analysis, line_box) or [line_box]
    span_length = int(random_source.integers(1, min(maximum_words, len(line_words)) + 1))
    span_start = int(random_source.integers(0, len(line_words) - span_length + 1))
    return union_box(line_words[span_start : span_start + span_length], line_box)


def plan_underlines(
    analysis: PageAnalysis,
    page_shape: tuple[int, int],
    drawing_tool: DrawingTool,
    random_source: numpy.random.Generator,
) -> list[LocalAlpha]:
    """Underline a phrase, or a run of consecutive lines (a passage).

    Args:
        analysis: Page analysis.
        page_shape: (height, width).
        drawing_tool: Tool of the mark.
        random_source: Random generator.

    Returns:
        Coverage of each underline.
    """
    if not analysis.line_boxes:
        return []
    run_length = int(random_source.integers(2, 7)) if random_source.random() < 0.35 else 1
    first_line = int(random_source.integers(0, max(1, len(analysis.line_boxes) - run_length + 1)))
    rendered_marks = []
    for line_index in range(first_line, min(len(analysis.line_boxes), first_line + run_length)):
        span_box = (
            analysis.line_boxes[line_index] if run_length > 1 else random_span(analysis, random_source, 7, line_index)
        )
        rendered_marks.append(
            render_polylines(underline_paths(span_box, random_source), drawing_tool.stroke_width, random_source)
        )
    return rendered_marks


def plan_span_mark(path_builder: Callable, maximum_words: int, width_factor: float) -> MarkPlanner:
    """Build a planner that draws ``path_builder`` around/through a random word span.

    Args:
        path_builder: ``underline_paths``-like function ``(span_box, random_source) -> polylines``.
        maximum_words: Maximum words in the span.
        width_factor: Multiplier of the tool width (highlighters use the line height instead).

    Returns:
        Mark planner.
    """

    def span_planner(
        analysis: PageAnalysis,
        page_shape: tuple[int, int],
        drawing_tool: DrawingTool,
        random_source: numpy.random.Generator,
    ) -> list[LocalAlpha]:
        """Render one mark around a random span; see ``plan_span_mark``."""
        span_box = random_span(analysis, random_source, maximum_words)
        if span_box is None:
            return []
        return [
            render_polylines(
                path_builder(span_box, random_source), drawing_tool.stroke_width * width_factor, random_source
            )
        ]

    return span_planner


def highlight_paths(span_box: PixelBox, random_source: numpy.random.Generator) -> list[numpy.ndarray]:
    """Center line of a highlighter band over a span.

    Args:
        span_box: Highlighted words.
        random_source: Random generator.

    Returns:
        One polyline along the vertical middle of the span.
    """
    middle_y = (span_box[1] + span_box[3]) / 2 + random_source.uniform(-1.5, 1.5)
    return [
        numpy.array(
            [
                [span_box[0] - random_source.uniform(0, 6), middle_y],
                [span_box[2] + random_source.uniform(0, 6), middle_y + random_source.uniform(-2, 2)],
            ],
            numpy.float32,
        )
    ]


def plan_margin_mark(
    analysis: PageAnalysis,
    page_shape: tuple[int, int],
    drawing_tool: DrawingTool,
    random_source: numpy.random.Generator,
) -> list[LocalAlpha]:
    """Bracket/bar beside a range of lines, or a tick/asterisk/arrow next to one line, in a margin.

    Args:
        analysis: Page analysis.
        page_shape: (height, width).
        drawing_tool: Tool of the mark.
        random_source: Random generator.

    Returns:
        Coverage of the mark.
    """
    if not analysis.line_boxes:
        return []
    text_box = analysis.text_box
    left_side = random_source.random() < 0.6
    margin_x = text_box[0] - random_source.uniform(5, 28) if left_side else text_box[2] + random_source.uniform(5, 28)
    first_line = int(random_source.integers(len(analysis.line_boxes)))
    if random_source.random() < 0.6:
        last_line = min(len(analysis.line_boxes) - 1, first_line + int(random_source.integers(1, 9)))
        polylines = bracket_paths(
            margin_x,
            analysis.line_boxes[first_line][1] - random_source.uniform(0, 5),
            analysis.line_boxes[last_line][3] + random_source.uniform(0, 5),
            left_side,
            random_source,
        )
    else:
        line_box = analysis.line_boxes[first_line]
        polylines = symbol_paths(margin_x, (line_box[1] + line_box[3]) / 2, random_source.uniform(7, 18), random_source)
    return [render_polylines(polylines, drawing_tool.stroke_width, random_source)]
