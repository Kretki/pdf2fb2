"""Runs a layout job against any detector that follows ``LayoutDetector``."""

from pathlib import Path
from typing import Protocol

from worker_protocol.job_schema import LayoutJob
from worker_protocol.progress_events import LAYOUT_PAGE_EVENT, emit_progress_event


class LayoutDetector(Protocol):
    """Anything that can detect layout boxes on a page image.

    Usage:
        Implemented by ``PaddleLayoutDetector``; tests may pass a fake.
    """

    def detect_boxes(self, image_path: Path, score_threshold: float) -> list[dict]:
        """Detect layout boxes on one page image.

        Args:
            image_path: Absolute path of the page image.
            score_threshold: Minimum detection score, 0..1.

        Returns:
            List of boxes ``{"label", "bbox", "score", "order"}`` in pixel coordinates.
        """
        ...


def run_layout_job(layout_job: LayoutJob, layout_detector: LayoutDetector) -> dict:
    """Segment every page of a layout job.

    Usage:
        Called by the worker entry point after the job file is validated. Emits a
        ``layout_page`` progress event before each page.

    Args:
        layout_job: Validated job with page images.
        layout_detector: Detector used for every page.

    Returns:
        Result document ``{"pages": [{"image_file": str, "boxes": [...]}, ...]}`` in page order.

    Raises:
        ValueError: If a page path escapes the working directory.
    """
    page_results = []
    page_count = len(layout_job.page_images)
    for page_number, image_file in enumerate(layout_job.page_images, start=1):
        emit_progress_event(LAYOUT_PAGE_EVENT, page=page_number, total=page_count)
        detected_boxes = layout_detector.detect_boxes(layout_job.resolve_inside(image_file), layout_job.score_threshold)
        page_results.append({"image_file": image_file, "boxes": detected_boxes})
        print(f"layout page {page_number}/{page_count}: {len(detected_boxes)} boxes", flush=True)
    return {"pages": page_results}
