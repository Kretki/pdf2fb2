"""Generates a small two-page PDF that exercises every converter feature.

Contents: title, numbered headings, a paragraph broken across pages, a footnote,
a picture with caption, a table, a running header and page numbers.
"""

import io
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw

PAGE_WIDTH = 595
PAGE_HEIGHT = 842
BODY_FONT_SIZE = 11
TABLE_ROWS = [["Model", "Score", "Params"], ["A", "91.2", "0.9B"], ["B", "88.4", "7B"], ["C", "85.0", "72B"]]


def _draw_running_header(pdf_page: pymupdf.Page, page_number: int) -> None:
    """Draw the running header and the printed page number.

    Args:
        pdf_page: Page to draw on.
        page_number: Number printed at the bottom.
    """
    pdf_page.insert_text((60, 40), "Running header - Test Book", fontsize=8)
    pdf_page.insert_text((290, 815), str(page_number), fontsize=10)


def _sample_picture_bytes() -> bytes:
    """Create a PNG with a circle and a rectangle.

    Returns:
        PNG-encoded image bytes.
    """
    sample_picture = Image.new("RGB", (400, 220), (235, 242, 250))
    drawing_context = ImageDraw.Draw(sample_picture)
    drawing_context.ellipse((40, 30, 200, 190), fill=(30, 144, 255))
    drawing_context.rectangle((230, 60, 360, 170), fill=(255, 140, 0))
    image_buffer = io.BytesIO()
    sample_picture.save(image_buffer, "PNG")
    return image_buffer.getvalue()


def _build_first_page(pdf_document: pymupdf.Document) -> None:
    """Add page 1: title, heading, footnoted paragraph, picture, caption, broken paragraph, footnote.

    Args:
        pdf_document: Document receiving the page.
    """
    first_page = pdf_document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    _draw_running_header(first_page, 1)
    first_page.insert_text((60, 90), "Test Document Title", fontsize=24)
    first_page.insert_text((60, 135), "1. Introduction", fontsize=16)
    first_page.insert_textbox(pymupdf.Rect(60, 150, 535, 240),
                              "This paragraph has a footnote marker here¹ and checks that body text is "
                              "segmented and read correctly by the OCR model. " * 2, fontsize=BODY_FONT_SIZE)
    first_page.insert_image(pymupdf.Rect(97, 260, 497, 480), stream=_sample_picture_bytes())
    first_page.insert_text((180, 500), "Figure 1. A circle and a rectangle", fontsize=10)
    first_page.insert_textbox(pymupdf.Rect(60, 530, 535, 700),
                              "The last paragraph of the first page is deliberately cut in the middle of a "
                              "sentence so that the converter has to join it with the text that continues on the",
                              fontsize=BODY_FONT_SIZE)
    first_page.draw_line((60, 760), (200, 760), width=0.5)
    first_page.insert_text((60, 778), "¹ This is the footnote text.", fontsize=8)


def _draw_results_table(pdf_page: pymupdf.Page) -> None:
    """Draw a 4x3 bordered table.

    Args:
        pdf_page: Page to draw on.
    """
    table_left, table_top, column_width, row_height = 60, 215, 158, 26
    for row_index, table_row in enumerate(TABLE_ROWS):
        for column_index, cell_text in enumerate(table_row):
            cell_rect = pymupdf.Rect(table_left + column_index * column_width, table_top + row_index * row_height,
                                     table_left + (column_index + 1) * column_width,
                                     table_top + (row_index + 1) * row_height)
            pdf_page.draw_rect(cell_rect, width=0.8)
            pdf_page.insert_text((cell_rect.x0 + 6, cell_rect.y1 - 8), cell_text, fontsize=BODY_FONT_SIZE)


def _build_second_page(pdf_document: pymupdf.Document) -> None:
    """Add page 2: paragraph continuation, heading, table caption, table, closing paragraph.

    Args:
        pdf_document: Document receiving the page.
    """
    second_page = pdf_document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    _draw_running_header(second_page, 2)
    second_page.insert_textbox(pymupdf.Rect(60, 70, 535, 130),
                               "next page, starting with a lowercase word, which must be merged.",
                               fontsize=BODY_FONT_SIZE)
    second_page.insert_text((60, 170), "2. Results", fontsize=16)
    second_page.insert_text((60, 205), "Table 1. Sample results", fontsize=10)
    _draw_results_table(second_page)
    second_page.insert_textbox(pymupdf.Rect(60, 350, 535, 450),
                               "Closing paragraph with a short summary of the results. " * 3, fontsize=BODY_FONT_SIZE)


def create_selftest_pdf(pdf_path: Path) -> Path:
    """Write the two-page self-test PDF.

    Usage:
        Called for ``--selftest``.

    Args:
        pdf_path: Destination file.

    Returns:
        ``pdf_path``.
    """
    with pymupdf.open() as pdf_document:
        _build_first_page(pdf_document)
        _build_second_page(pdf_document)
        pdf_document.set_metadata({"title": "Test Book", "author": "Jane Tester"})
        pdf_document.save(pdf_path)
    return pdf_path
