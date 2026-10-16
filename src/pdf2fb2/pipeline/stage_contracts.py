"""Abstractions of the model stages the conversion pipeline depends on.

A page record is a dict ``{"page": int (1-based), "image": str (path),
"size": [width, height], "segments": [segment, ...]}``; a segment is a dict with
``label``, ``bbox``, ``score``, ``order`` and, after OCR, ``text``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pdf2fb2.progress.progress_contracts import StageProgress


class PageCleaner(Protocol):
    """Removes reader marks (handwriting, pen/pencil marks) from already segmented and OCR'd pages."""

    def clean_pages(self, page_records: list[dict], work_directory: Path,
                    stage_progress: StageProgress) -> dict[int, Path]:
        """Detect reader marks with picture segments blanked out, and clean the pages that have them.

        Args:
            page_records: Segmented page records (picture segments are blanked before detection).
            work_directory: Output directory used for job, result and per-page mark files.
            stage_progress: Receives per-page progress.

        Returns:
            Mapping page number -> cleaned page image, only for pages with reader marks.
        """
        ...


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


@dataclass(frozen=True)
class RescanProgress:
    """Progress sinks of the three steps of the reader-mark rescan.

    Attributes:
        cleaning_progress: Reader-mark detection and cleaning.
        segmentation_progress: Layout of the cleaned pages.
        recognition_progress: OCR of the cleaned pages.
    """

    cleaning_progress: StageProgress
    segmentation_progress: StageProgress
    recognition_progress: StageProgress


class MarkedPageRescanner(Protocol):
    """Cleans pages with reader marks and replaces their records by a rescan of the cleaned page."""

    def rescan_marked_pages(self, page_records: list[dict], work_directory: Path,
                            rescan_progress: RescanProgress) -> list[dict]:
        """Clean marked pages, segment and OCR them again, and merge them with the untouched pages.

        Args:
            page_records: OCR'd page records of the first pass.
            work_directory: Conversion output directory.
            rescan_progress: Progress sinks of the three steps.

        Returns:
            Final page records in page order (rescanned pages replaced, pictures kept from the first pass).
        """
        ...
