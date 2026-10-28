"""Pages of a book used by the benchmark, and the small PDF the pipeline converts instead of the whole book.

The sample is one run of consecutive pages, placed reproducibly (seeded by the book key) in the body of the
book, skipping the first and last 8% (covers, title pages, indexes), so every run of the same split reads the
same pages. Consecutive pages keep page breaks real: a paragraph running over a page break is joined with its
true continuation, as in a whole-book conversion.
"""

import hashlib
import random
from pathlib import Path

import pymupdf

BODY_MARGIN = 0.08


def sample_pages(book_key: str, page_count: int, sample_size: int) -> list[int]:
    """0-based page indexes of the sample (consecutive, in reading order).

    Args:
        book_key: Key of the book (seed).
        page_count: Pages of the scan.
        sample_size: Pages wanted (all body pages when the book is shorter).

    Returns:
        Page indexes in reading order.
    """
    first_page = int(page_count * BODY_MARGIN)
    body_pages = list(range(first_page, max(first_page + 1, page_count - first_page)))
    seed_value = int(hashlib.sha256(book_key.encode("utf-8")).hexdigest()[:16], 16)
    sample_generator = random.Random(seed_value)  # nosec B311  # noqa: S311 - reproducible sampling, not cryptography
    run_length = min(sample_size, len(body_pages))
    run_start = sample_generator.randrange(len(body_pages) - run_length + 1)
    return body_pages[run_start:run_start + run_length]


def write_page_subset(pdf_path: Path, page_indexes: list[int], subset_path: Path) -> None:
    """Copy the sampled pages into a new PDF.

    Args:
        pdf_path: Full scan as PDF.
        page_indexes: 0-based pages to copy, in order.
        subset_path: Output PDF (overwritten); its page n is ``page_indexes[n - 1]``.

    Returns:
        None.
    """
    subset_path.parent.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(pdf_path) as source_document, pymupdf.open() as subset_document:
        for page_index in page_indexes:
            subset_document.insert_pdf(source_document, from_page=page_index, to_page=page_index)
        subset_document.save(subset_path)
