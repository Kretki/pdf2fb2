"""Converts a list of books one after another and prints one summary at the end.

A failing book (missing converter, worker failure, unreadable file) is reported in the summary and the next book
is converted; the exit code is 1 when any book failed or did not pass its checks.
"""

import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import TextIO

from pdf2fb2.document_sources.source_formats import DocumentConversionError
from pdf2fb2.interface.book_jobs import BookConversionJob
from pdf2fb2.worker_runtime.worker_launcher import WorkerFailedError

SUCCESS_EXIT_CODE = 0
FAILURE_EXIT_CODE = 1
HANDLED_ERRORS = (WorkerFailedError, DocumentConversionError, OSError, ValueError)
STATUS_OK = "ok"
STATUS_CHECKS = "checks failed"
STATUS_ERROR = "error"


@dataclass(frozen=True)
class BookResult:
    """Outcome of one book.

    Attributes:
        book_job: The converted book.
        status_text: ``ok``, ``checks failed`` or ``error``.
        error_text: Error message for ``error`` (on one line), else empty.
    """

    book_job: BookConversionJob
    status_text: str
    error_text: str = ""


def convert_one_book(book_job: BookConversionJob, convert_book: Callable[[BookConversionJob], int]) -> BookResult:
    """Convert one book, turning the expected failures into a result.

    Args:
        book_job: Book and output directory.
        convert_book: Runs the conversion and returns its exit code (0: all checks passed).

    Returns:
        The book's result.
    """
    try:
        exit_code = convert_book(book_job)
    except HANDLED_ERRORS as conversion_error:
        return BookResult(book_job, STATUS_ERROR, " ".join(str(conversion_error).split()))
    return BookResult(book_job, STATUS_OK if exit_code == SUCCESS_EXIT_CODE else STATUS_CHECKS)


def print_summary(book_results: list[BookResult], output_stream: TextIO) -> None:
    """Print one line per book and the totals.

    Args:
        book_results: Results in conversion order.
        output_stream: Stream the summary is written to.

    Returns:
        None.
    """
    status_counts = {status_text: sum(book_result.status_text == status_text for book_result in book_results)
                     for status_text in (STATUS_OK, STATUS_CHECKS, STATUS_ERROR)}
    print(f"\nBooks: {len(book_results)}  " + ", ".join(f"{status_text}: {status_count}"
                                                      for status_text, status_count in status_counts.items()),
          file=output_stream)
    for book_result in book_results:
        error_suffix = f"  ({book_result.error_text})" if book_result.error_text else ""
        print(f"  {book_result.status_text:<13} {book_result.book_job.book_path} -> "
              f"{book_result.book_job.output_directory}{error_suffix}", file=output_stream)


def convert_books(book_jobs: list[BookConversionJob], convert_book: Callable[[BookConversionJob], int],
                  output_stream: TextIO = sys.stdout) -> int:
    """Convert every book and print the summary.

    Usage:
        Called by the command line for ``--input-dir`` runs.

    Args:
        book_jobs: Books to convert, in order.
        convert_book: Runs one conversion and returns its exit code.
        output_stream: Stream for the summary.

    Returns:
        0 when every book converted and passed its checks (or there were no books), else 1.
    """
    book_results = [convert_one_book(book_job, convert_book) for book_job in book_jobs]
    print_summary(book_results, output_stream)
    return SUCCESS_EXIT_CODE if all(book_result.status_text == STATUS_OK for book_result in book_results) \
        else FAILURE_EXIT_CODE
