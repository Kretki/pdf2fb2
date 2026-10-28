"""Page count of an input book without converting it, for the start-of-run time estimate.

PDF pages are counted with PyMuPDF; DjVu pages are read from the file header (the ``DIRM`` directory of a
multi-page document counts its component files, which may include a few shared ones: the count errs high).
"""

from collections.abc import Callable
from pathlib import Path

from pdf2fb2.rendering.pdf_rendering import count_pdf_pages

DJVU_HEADER_BYTES = 32
DJVU_MAGIC = b"AT&TFORM"
SINGLE_PAGE_FORM = b"DJVU"
MULTI_PAGE_FORM = b"DJVM"
DIRECTORY_CHUNK = b"DIRM"
FORM_TYPE_SLICE = slice(12, 16)
CHUNK_NAME_SLICE = slice(16, 20)
FILE_COUNT_SLICE = slice(25, 27)
PageCounter = Callable[[Path], int]


def count_djvu_pages(djvu_path: Path) -> int:
    """Number of pages of a DjVu file, from its header.

    Args:
        djvu_path: Input DjVu file.

    Returns:
        1 for a single-page document, else the number of files in its directory (>= the page count).

    Raises:
        ValueError: If the header is not a DjVu header.
        OSError: If the file cannot be read.
    """
    with djvu_path.open("rb") as djvu_file:
        header_bytes = djvu_file.read(DJVU_HEADER_BYTES)
    if not header_bytes.startswith(DJVU_MAGIC):
        raise ValueError(f"not a DjVu file: {djvu_path.name}")
    form_type = header_bytes[FORM_TYPE_SLICE]
    if form_type == SINGLE_PAGE_FORM:
        return 1
    if form_type != MULTI_PAGE_FORM or header_bytes[CHUNK_NAME_SLICE] != DIRECTORY_CHUNK:
        raise ValueError(f"unknown DjVu layout: {djvu_path.name}")
    return int.from_bytes(header_bytes[FILE_COUNT_SLICE], "big")


PAGE_COUNTERS: dict[str, PageCounter] = {".pdf": count_pdf_pages, ".djvu": count_djvu_pages,
                                         ".djv": count_djvu_pages}


def count_book_pages(book_path: Path, page_counters: dict[str, PageCounter] = PAGE_COUNTERS) -> int | None:
    """Page count of a book of any supported format.

    Usage:
        Called before a run to estimate its duration; never raises for unreadable books.

    Args:
        book_path: Input book.
        page_counters: Counter per lowercase suffix.

    Returns:
        Page count, or None for an unknown format or an unreadable file.
    """
    page_counter = page_counters.get(book_path.suffix.lower())
    if page_counter is None:
        return None
    try:
        return page_counter(book_path)
    except (OSError, ValueError, RuntimeError):
        return None
