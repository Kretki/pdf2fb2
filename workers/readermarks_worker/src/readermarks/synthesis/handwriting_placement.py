"""Handwriting planners: small notes between/over printed lines and notes in the margins."""

import numpy

from readermarks.synthesis.ink_compositing import DrawingTool
from readermarks.synthesis.mark_placement import random_span
from readermarks.synthesis.note_vocabulary import note_text
from readermarks.synthesis.page_analysis import PageAnalysis
from readermarks.synthesis.stroke_rendering import LocalAlpha, render_handwriting


def handwriting_alpha(
    analysis: PageAnalysis,
    drawing_tool: DrawingTool,
    random_source: numpy.random.Generator,
    letter_range: tuple[float, float],
    maximum_words: int,
    rotation_degrees: float,
) -> numpy.ndarray:
    """Render a note with the tool's width and a letter height relative to the print size.

    Args:
        analysis: Page analysis (``x_height`` scales the letters).
        drawing_tool: Tool of the note.
        random_source: Random generator.
        letter_range: Letter height range as multiples of the printed x-height.
        maximum_words: Maximum words in the note.
        rotation_degrees: Rotation of the note.

    Returns:
        (h, w) float32 coverage.
    """
    letter_height = analysis.x_height * random_source.uniform(*letter_range)
    stroke_width = min(drawing_tool.stroke_width * random_source.uniform(0.9, 1.4), 0.15 * letter_height + 1.0)
    return render_handwriting(
        note_text(random_source, maximum_words),
        letter_height,
        stroke_width,
        rotation_degrees + random_source.uniform(-6, 6),
        random_source,
    )


def plan_inline_text(
    analysis: PageAnalysis,
    page_shape: tuple[int, int],
    drawing_tool: DrawingTool,
    random_source: numpy.random.Generator,
) -> list[LocalAlpha]:
    """Small handwriting between two printed lines or over a word.

    Args:
        analysis: Page analysis.
        page_shape: (height, width).
        drawing_tool: Tool of the note.
        random_source: Random generator.

    Returns:
        Coverage of the note.
    """
    span_box = random_span(analysis, random_source, 4)
    if span_box is None:
        return []
    note_alpha = handwriting_alpha(analysis, drawing_tool, random_source, (0.6, 1.3), 3, 0.0)
    vertical_shift = random_source.uniform(-0.9, 0.2) * note_alpha.shape[0]
    return [
        LocalAlpha(note_alpha, int(span_box[0] + random_source.uniform(-10, 20)), int(span_box[1] + vertical_shift))
    ]


def plan_margin_text(
    analysis: PageAnalysis,
    page_shape: tuple[int, int],
    drawing_tool: DrawingTool,
    random_source: numpy.random.Generator,
) -> list[LocalAlpha]:
    """A handwritten note in the top/bottom or side margin (side notes are often vertical).

    Args:
        analysis: Page analysis.
        page_shape: (height, width).
        drawing_tool: Tool of the note.
        random_source: Random generator.

    Returns:
        Coverage of the note.
    """
    page_height, page_width = page_shape
    text_box = analysis.text_box
    side_draw = random_source.random()
    vertical_note = side_draw < 0.5 and random_source.random() < 0.45
    note_alpha = handwriting_alpha(
        analysis,
        drawing_tool,
        random_source,
        (0.9, 2.6),
        4,
        float(random_source.choice([90.0, -90.0])) if vertical_note else 0.0,
    )
    note_height, note_width = note_alpha.shape
    if side_draw < 0.5:
        center_x = text_box[0] / 2 if random_source.random() < 0.5 else (text_box[2] + page_width) / 2
        x_offset = int(center_x - note_width / 2 + random_source.uniform(-10, 10))
        y_offset = int(random_source.uniform(text_box[1], max(text_box[1] + 1, text_box[3] - note_height)))
    else:
        top_side = side_draw < 0.8
        center_y = text_box[1] / 2 if top_side else (text_box[3] + page_height) / 2
        y_offset = int(center_y - note_height / 2 + random_source.uniform(-8, 8))
        x_offset = int(random_source.uniform(text_box[0], max(text_box[0] + 1, text_box[2] - note_width)))
    return [LocalAlpha(note_alpha, x_offset, y_offset)]
