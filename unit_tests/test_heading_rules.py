"""Unit tests of the running-head band check (before OCR) and the opening-heading rule (after OCR)."""

from pdf2fb2.segments.opening_headings import mark_opening_headings
from pdf2fb2.segments.running_heads import relabel_misplaced_running_heads

PAGE_WIDTH = 834
PAGE_HEIGHT = 1425
BODY_PARAGRAPH = "word " * 60


def text_segment(segment_box: list[float], segment_text: str, segment_label: str = "text") -> dict:
    """OCR'd segment.

    Args:
        segment_box: [x1, y1, x2, y2] in pixels.
        segment_text: OCR text.
        segment_label: Layout label.

    Returns:
        Segment dictionary.
    """
    return {"label": segment_label, "bbox": segment_box, "score": 0.9, "text": segment_text, "glyph_height": 14.0}


def body_page(page_number: int, extra_segments: list[dict], body_top: float = 100.0) -> dict:
    """Page with a running head, two body paragraphs and a page number.

    Args:
        page_number: Page number (its parity selects the page group).
        extra_segments: Segments added at the start.
        body_top: Top of the first paragraph in pixels.

    Returns:
        Page record.
    """
    return {"page": page_number, "size": [PAGE_WIDTH, PAGE_HEIGHT], "segments": [
        *extra_segments,
        text_segment([60, body_top, 770, body_top + 500], BODY_PARAGRAPH),
        text_segment([60, body_top + 501, 770, 1190], BODY_PARAGRAPH),
        text_segment([400, 1215, 420, 1240], "1", "number")]}


def test_header_far_from_the_running_head_band_becomes_a_heading() -> None:
    """A chapter title labelled ``header`` a fifth down the page is relabelled; real running heads stay."""
    page_records = [body_page(page_number, [text_segment([300, 55, 700, 90], "", "header")])
                    for page_number in range(2, 20)]
    page_records[3]["segments"][0]["bbox"] = [112, 268, 691, 306]
    assert relabel_misplaced_running_heads(page_records) == 1
    assert page_records[3]["segments"][0]["label"] == "paragraph_title"
    assert page_records[3]["segments"][0]["original_label"] == "header"
    assert page_records[4]["segments"][0]["label"] == "header"


def test_title_line_set_apart_above_the_first_paragraph_becomes_a_heading() -> None:
    """A short centred line 57 px above the text is a heading; a short line close to the text is not."""
    page_records = [body_page(page_number, []) for page_number in range(2, 20)]
    page_records[2] = body_page(4, [text_segment([222, 270, 580, 301], "EDITOR'S PREFACE")], body_top=358.0)
    page_records[5] = body_page(7, [text_segment([222, 270, 580, 301], "A short line")], body_top=310.0)
    assert mark_opening_headings(page_records) == 1
    assert page_records[2]["segments"][0]["label"] == "paragraph_title"
    assert page_records[5]["segments"][0]["label"] == "text"
