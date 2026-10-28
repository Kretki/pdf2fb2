"""Start-of-run estimate and admin confirmation of conversions and benchmark runs (wiring only).

Usage:
    ``confirm_book_jobs`` is called by ``command_line`` before converting, ``confirm_benchmark_run`` by
    ``pdf2fb2-bench run`` before its full cycle; ``--yes`` skips the question (the estimate is still printed).
"""

import argparse
from pathlib import Path

from pdf2fb2.document_sources.page_counting import count_book_pages
from pdf2fb2.interface.book_jobs import BookConversionJob
from pdf2fb2.interface.pipeline_wiring import build_pipeline
from pdf2fb2.interface.start_confirmation import confirm_start
from pdf2fb2.progress.terminal_progress import SilentProgressDisplay
from pdf2fb2.project_paths import SPEED_METRICS_FILE
from pdf2fb2.timekeeping.run_estimate import describe_estimate, estimate_books
from pdf2fb2.timekeeping.speed_metrics_store import JsonSpeedMetricsStore
from pdf2fb2.timekeeping.time_estimation import ConversionPlan


def plan_for_arguments(parsed_arguments: argparse.Namespace) -> ConversionPlan:
    """Stages and proofreading scope the parsed options run.

    Args:
        parsed_arguments: Parsed ``pdf2fb2`` command line.

    Returns:
        The conversion plan (no worker is started).
    """
    return build_pipeline(parsed_arguments, SilentProgressDisplay()).conversion_plan(
        not parsed_arguments.skip_drawings)


def confirm_estimate(conversion_plan: ConversionPlan, book_pages: list[tuple[str, int | None]],
                     ask_admin: bool) -> bool:
    """Print the estimate of a run and ask the admin to start or abort it.

    Args:
        conversion_plan: Stages and proofreading scope of every conversion.
        book_pages: ``(label, pages to convert or None)`` per book.
        ask_admin: False prints the estimate only and starts at once.

    Returns:
        True to start, False to abort.
    """
    speed_metrics = JsonSpeedMetricsStore(SPEED_METRICS_FILE).load_metrics()
    estimate_lines = describe_estimate(conversion_plan, estimate_books(conversion_plan, book_pages, speed_metrics),
                                       speed_metrics)
    if not ask_admin:
        print("\n".join(estimate_lines))
        return True
    return confirm_start(estimate_lines)


def confirm_book_jobs(parsed_arguments: argparse.Namespace, book_jobs: list[BookConversionJob]) -> bool:
    """Estimate the books of a ``pdf2fb2`` run and ask the admin.

    Args:
        parsed_arguments: Parsed command line (``--yes`` skips the question).
        book_jobs: Books to convert.

    Returns:
        True to start, False to abort.
    """
    book_pages = [(book_job.book_path.name, count_book_pages(book_job.book_path)) for book_job in book_jobs]
    return confirm_estimate(plan_for_arguments(parsed_arguments), book_pages,
                            not parsed_arguments.skip_confirmation)


def sampled_page_count(scan_path: Path, pages_per_book: int) -> int | None:
    """Pages a benchmark converts of one book.

    Args:
        scan_path: Scan of the book.
        pages_per_book: Sampled pages per book.

    Returns:
        ``min(pages_per_book, page count)``, or None when the scan cannot be counted.
    """
    page_count = count_book_pages(scan_path)
    return None if page_count is None else min(page_count, pages_per_book)


def confirm_benchmark_run(parsed_pipeline_arguments: argparse.Namespace, benchmark_books: list[tuple[str, Path]],
                          pages_per_book: int, ask_admin: bool) -> bool:
    """Estimate the full cycle of a benchmark run (every sampled book) and ask the admin.

    Args:
        parsed_pipeline_arguments: Parsed ``pdf2fb2`` options of the conversions.
        benchmark_books: ``(label, scan path)`` of every book to convert (reused conversions excluded).
        pages_per_book: Sampled pages per book.
        ask_admin: False prints the estimate only.

    Returns:
        True to start, False to abort.
    """
    book_pages = [(book_label, sampled_page_count(scan_path, pages_per_book))
                  for book_label, scan_path in benchmark_books]
    return confirm_estimate(plan_for_arguments(parsed_pipeline_arguments), book_pages, ask_admin)
