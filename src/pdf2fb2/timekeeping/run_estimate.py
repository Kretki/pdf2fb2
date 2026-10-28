"""Start-of-run estimate of one or more conversions, and its text for the admin.

Every page is taken for a text-only page with the stored (upper) text density and mistake share, so the
estimate errs on the long side: books with pictures or few mistakes finish earlier.
"""

from dataclasses import dataclass

from pdf2fb2.progress.progress_formatting import format_duration
from pdf2fb2.timekeeping.speed_metrics import SpeedMetrics
from pdf2fb2.timekeeping.time_estimation import ConversionPlan, remaining_run_seconds
from pdf2fb2.timekeeping.workload_units import ConversionWorkload

UNKNOWN_PAGES_TEXT = "? pages"


@dataclass(frozen=True)
class BookEstimate:
    """Estimate of one book.

    Attributes:
        book_label: Name shown to the admin.
        page_count: Pages to convert (None: unknown, not counted in the total).
        estimated_seconds: Estimated duration (None when the page count is unknown).
    """

    book_label: str
    page_count: int | None
    estimated_seconds: float | None


def estimate_book_seconds(conversion_plan: ConversionPlan, page_count: int, speed_metrics: SpeedMetrics) -> float:
    """Estimated duration of one conversion before it starts.

    Args:
        conversion_plan: Active stages and proofreading scope.
        page_count: Pages to convert (>= 0).
        speed_metrics: Stored speeds of this machine.

    Returns:
        Seconds (>= 0).
    """
    return remaining_run_seconds(conversion_plan, ConversionWorkload(page_count=page_count), speed_metrics, {})


def estimate_books(conversion_plan: ConversionPlan, book_pages: list[tuple[str, int | None]],
                   speed_metrics: SpeedMetrics) -> list[BookEstimate]:
    """Estimate every book of a run.

    Args:
        conversion_plan: Active stages and proofreading scope.
        book_pages: ``(label, page count or None)`` per book.
        speed_metrics: Stored speeds of this machine.

    Returns:
        Estimates in input order.
    """
    return [BookEstimate(book_label, page_count, None if page_count is None
                         else estimate_book_seconds(conversion_plan, page_count, speed_metrics))
            for book_label, page_count in book_pages]


def describe_book(book_estimate: BookEstimate) -> str:
    """One summary line of a book.

    Args:
        book_estimate: Estimate of the book.

    Returns:
        ``  <label>  <pages> pages  ~<duration>``.
    """
    pages_text = UNKNOWN_PAGES_TEXT if book_estimate.page_count is None else f"{book_estimate.page_count} pages"
    duration_text = "?" if book_estimate.estimated_seconds is None else format_duration(book_estimate.estimated_seconds)
    return f"  {book_estimate.book_label}  {pages_text}  ~{duration_text}"


def describe_estimate(conversion_plan: ConversionPlan, book_estimates: list[BookEstimate],
                      speed_metrics: SpeedMetrics) -> list[str]:
    """Lines of the start-of-run estimate.

    Usage:
        Printed before asking the admin to start or abort.

    Args:
        conversion_plan: Active stages and proofreading scope.
        book_estimates: Estimate of every book.
        speed_metrics: Stored speeds (``run_count`` tells how much is measured).

    Returns:
        Text lines without newlines.
    """
    known_estimates = [book_estimate for book_estimate in book_estimates if book_estimate.estimated_seconds is not None]
    total_pages = sum(book_estimate.page_count or 0 for book_estimate in known_estimates)
    total_seconds = sum(book_estimate.estimated_seconds or 0.0 for book_estimate in known_estimates)
    speed_source = f"speeds measured over {speed_metrics.run_count} runs" if speed_metrics.run_count \
        else "built-in speeds (no run measured on this machine yet)"
    return [f"Time estimate ({speed_source}, every page taken for a text page):",
            f"  stages: {', '.join(conversion_plan.stage_names)}; proofreading scope: "
            f"{conversion_plan.proofread_scope}",
            *(describe_book(book_estimate) for book_estimate in book_estimates),
            f"Total: {len(book_estimates)} books, {total_pages} pages, ~{format_duration(total_seconds)}"]
