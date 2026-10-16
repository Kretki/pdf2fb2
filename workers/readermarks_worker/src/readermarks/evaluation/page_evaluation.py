"""Page-level evaluation on real pages: has_marks accuracy, unmarked-page fidelity, speed and VRAM."""

import time

import numpy
import torch

from readermarks.datasets.layout_regions import read_masked_page
from readermarks.datasets.page_inventory import RealPage
from readermarks.evaluation.classification_metrics import DetectionCounts
from readermarks.evaluation.patch_evaluation import mean_or_none
from readermarks.inference.page_cleaner import PageCleaner


def evaluate_real_pages(
    page_cleaner: PageCleaner, real_pages: list[RealPage], picture_regions: dict[int, list[list[float]]]
) -> dict:
    """Run the full pipeline on labeled real pages and summarise.

    Usage:
        Called by the evaluation service for ``real_val`` or ``real_test``.

    Args:
        page_cleaner: Loaded cleaner with effective thresholds.
        real_pages: Labeled pages (``has_marks`` is not None).
        picture_regions: Picture boxes per page, blanked before inspection (as in pdf2fb2).

    Returns:
        ``{"page_level", "pages", "unmarked_altered", "timings", "peak_vram_gb"}``.
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    page_counts = DetectionCounts()
    page_rows, marked_seconds, unmarked_seconds, unmarked_differences = [], [], [], []
    for real_page in real_pages:
        rgb_image = read_masked_page(real_page, picture_regions)
        start_time = time.perf_counter()
        page_verdict = page_cleaner.inspect_page(rgb_image)
        elapsed_seconds = time.perf_counter() - start_time
        (marked_seconds if page_verdict.has_marks else unmarked_seconds).append(elapsed_seconds)
        page_counts.add_arrays(numpy.array([page_verdict.has_marks]), numpy.array([bool(real_page.has_marks)]))
        output_image = page_verdict.cleaned_image if page_verdict.has_marks else rgb_image
        mean_difference = float(numpy.abs(output_image.astype(numpy.int16) - rgb_image).mean())
        if not real_page.has_marks:
            unmarked_differences.append(mean_difference)
        page_rows.append(
            {
                "page": real_page.page_number,
                "expected": bool(real_page.has_marks),
                "predicted": page_verdict.has_marks,
                "page_probability": round(page_verdict.spotter_output.page_probability, 4),
                "regions": len(page_verdict.region_boxes),
                "marks": len(page_verdict.fused_marks.components) if page_verdict.has_marks else 0,
                "mean_abs_difference": round(mean_difference, 4),
                "seconds": round(elapsed_seconds, 3),
            }
        )
    return {
        "page_level": page_counts.summarize_counts(),
        "pages": page_rows,
        "unmarked_altered": sum(difference > 0 for difference in unmarked_differences),
        "unmarked_mean_abs_difference": mean_or_none(unmarked_differences),
        "timings": {
            "marked_page_seconds": mean_or_none(marked_seconds),
            "unmarked_page_seconds": mean_or_none(unmarked_seconds),
        },
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if torch.cuda.is_available() else None,
    }
