"""Abstractions of the model stages the conversion pipeline depends on.

A page record is a dict ``{"page": int (1-based), "image": str (path),
"size": [width, height], "segments": [segment, ...]}``; a segment is a dict with
``label``, ``bbox``, ``score``, ``order`` and, after OCR, ``text``.
"""

from pathlib import Path
from typing import Protocol

from pdf2fb2.progress_contracts import StageProgress


class PageSegmenter(Protocol):
    """Turns rendered page images into page records with ordered segments."""

    def segment_pages(self, page_images: list[Path], work_directory: Path,
                      stage_progress: StageProgress) -> list[dict]:
        """Segment all pages.

        Args:
            page_images: Rendered page images inside ``work_directory``, in page order.
            work_directory: Output directory used for job and result files.
            stage_progress: Receives per-page progress.

        Returns:
            One page record per image, in page order.
        """
        ...


class SegmentRecognizer(Protocol):
    """Fills the ``text`` field of every segment that needs OCR."""

    def recognize_segments(self, page_records: list[dict], work_directory: Path,
                           stage_progress: StageProgress) -> None:
        """Read the text of all OCR-able segments.

        Args:
            page_records: Page records from a ``PageSegmenter`` (mutated in place).
            work_directory: Output directory used for crops, cache, job and result files.
            stage_progress: Receives per-segment progress and failure messages.

        Returns:
            None.
        """
        ...
