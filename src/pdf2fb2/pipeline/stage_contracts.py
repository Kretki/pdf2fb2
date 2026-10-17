"""Abstractions of the model stages the conversion pipeline depends on.

A page record is a dict ``{"page": int (1-based), "image": str (path),
"size": [width, height], "segments": [segment, ...]}``; a segment is a dict with
``label``, ``bbox``, ``score``, ``order`` and, after OCR, ``text``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pdf2fb2.fb2_output.flow_model import BookFlow
from pdf2fb2.progress.progress_contracts import StageProgress
from pdf2fb2.segments.font_statistics import PageGlyphs


class PageCleaner(Protocol):
    """Removes reader marks (handwriting, pen/pencil marks) from segmented pages before OCR."""

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


@dataclass(frozen=True)
class IsolatedRegion:
    """A page region checked for reader marks on its own, with the rest of the page blanked.

    Attributes:
        page_image: Rendered page image inside the work directory.
        page_size: (width, height) of the page image in pixels.
        region_box: [x1, y1, x2, y2] of the region in page pixels.
    """

    page_image: Path
    page_size: tuple[int, int]
    region_box: list[float]


class RegionMarkInspector(Protocol):
    """Tells which page regions contain reader marks (handwriting, pen/pencil marks)."""

    def find_marked_regions(self, isolated_regions: list[IsolatedRegion], work_directory: Path,
                            stage_progress: StageProgress) -> set[int]:
        """Run reader-mark detection on every region with the rest of its page blanked.

        Args:
            isolated_regions: Regions to inspect (at least one).
            work_directory: Directory used for staged images, job, result and per-region mark files.
            stage_progress: Receives per-region progress.

        Returns:
            Indexes into ``isolated_regions`` of the regions with reader marks.
        """
        ...


class HandwritingDetector(Protocol):
    """Marks segments that are human-written notes instead of book text."""

    def mark_handwritten_segments(self, page_records: list[dict], work_directory: Path,
                                  stage_progress: StageProgress) -> int:
        """Find handwritten segments in the segmented pages and relabel them.

        Args:
            page_records: Segmented page records of the first layout pass (mutated: labels rewritten).
            work_directory: Conversion output directory.
            stage_progress: Receives the check's progress.

        Returns:
            Number of segments marked as handwritten.
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


class BoxProposer(Protocol):
    """Proposes every layout box of a page above a (low) score threshold, overlapping ones included."""

    def propose_boxes(self, page_images: list[Path], work_directory: Path,
                      stage_progress: StageProgress) -> list[list[dict]]:
        """Detect the layout boxes of the pages.

        Args:
            page_images: Page images inside ``work_directory``, in page order.
            work_directory: Output directory used for job and result files.
            stage_progress: Receives per-page progress.

        Returns:
            Per page, boxes ``{"label", "bbox", "score", "order"}``.
        """
        ...


class GlyphMeter(Protocol):
    """Measures the lettering of a segmented page."""

    def measure_page(self, page_record: dict) -> PageGlyphs:
        """Measure the glyph heights of the segments and find letter-like ink outside them.

        Args:
            page_record: Page record with ``image`` and ``segments``.

        Returns:
            Measurements aligned with the page's segments.
        """
        ...


class PictureRecoverer(Protocol):
    """Turns text segments that are really lettering of a picture into that picture."""

    def recover_pictures(self, page_records: list[dict], work_directory: Path,
                         stage_progress: StageProgress) -> int:
        """Find lettering that does not fit the book's font size and replace it by the picture it belongs to.

        Args:
            page_records: Final segmented page records, not yet OCR'd (mutated).
            work_directory: Conversion output directory.
            stage_progress: Receives the step's progress.

        Returns:
            Number of recovered pictures.
        """
        ...


class ResidualTextFinder(Protocol):
    """Adds segments for text the layout passes left outside every segment."""

    def add_residual_text(self, page_records: list[dict], work_directory: Path, stage_progress: StageProgress) -> int:
        """Find letters of the usual font size outside every segment and add segments for them.

        Args:
            page_records: Final segmented page records, not yet OCR'd (mutated).
            work_directory: Conversion output directory.
            stage_progress: Receives the step's progress.

        Returns:
            Number of added segments.
        """
        ...


class LayoutReviewer(Protocol):
    """Corrects segment labels once the text of the segments is known."""

    def review_layout(self, page_records: list[dict], stage_progress: StageProgress) -> int:
        """Apply the label corrections to the OCR'd pages.

        Args:
            page_records: OCR'd page records (mutated).
            stage_progress: Receives the step's progress.

        Returns:
            Number of relabelled segments.
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


class TextStyleDetector(Protocol):
    """Detects word-level text attributes (bold, italic, underline, strikeout) inside text segments."""

    def detect_text_styles(self, page_records: list[dict], work_directory: Path,
                           stage_progress: StageProgress) -> None:
        """Store the detected words with their attributes in the ``words`` field of the text segments.

        Args:
            page_records: Final segmented page records (mutated).
            work_directory: Output directory used for job and result files.
            stage_progress: Receives per-page progress.

        Returns:
            None.
        """
        ...


@dataclass(frozen=True)
class RescanProgress:
    """Progress sinks of the two steps of the reader-mark rescan.

    Attributes:
        cleaning_progress: Reader-mark detection and cleaning.
        segmentation_progress: Layout of the cleaned pages.
    """

    cleaning_progress: StageProgress
    segmentation_progress: StageProgress


class MarkedPageRescanner(Protocol):
    """Cleans pages with reader marks and replaces their records by a re-segmentation of the cleaned page."""

    def rescan_marked_pages(self, page_records: list[dict], work_directory: Path,
                            rescan_progress: RescanProgress) -> list[dict]:
        """Clean marked pages, segment them again, and merge them with the untouched pages (no OCR).

        Args:
            page_records: Segmented page records of the first pass (not yet OCR'd).
            work_directory: Conversion output directory.
            rescan_progress: Progress sinks of the two steps.

        Returns:
            Final segmented page records in page order (rescanned pages replaced, pictures kept from the
            first pass), ready for OCR.
        """
        ...


class FlowChecker(Protocol):
    """Proofreads the book flow and rearranges it (removes text that does not belong to the book)."""

    def proofread_flow(self, book_flow: BookFlow, work_directory: Path, stage_progress: StageProgress) -> dict:
        """Check every token of the flow text and apply the decided edits.

        Args:
            book_flow: Built flow (mutated).
            work_directory: Output directory for job, result and report files.
            stage_progress: Receives proofreading progress.

        Returns:
            Summary counts of the edits.
        """
        ...
