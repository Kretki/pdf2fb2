"""Handwriting check of position outliers: runs after the first layout pass, before whole-page mark cleaning.

Flow:
    1. text segments protruding above or below the printed frame of the book (see ``segments.frame_statistics``)
       are relabelled ``handwritten`` at once: nothing printed lies there, and the reader-mark model is not
       trained on hand-lettered words such as a ``NOT TRUE!`` written over the running head;
    2. segments whose horizontal position lies outside the 2-sigma interval of their group are found
       (see ``segments.position_statistics``);
    3. each of them is inspected by the reader-mark model alone (the rest of its page blanked), with job
       files under ``<output>/outliers``;
    4. outliers with reader marks are relabelled ``handwritten``: they are neither OCR'd nor put into the book.
"""

from pathlib import Path

from pdf2fb2.pipeline.stage_contracts import IsolatedRegion, RegionMarkInspector
from pdf2fb2.progress.progress_contracts import StageProgress
from pdf2fb2.segments.frame_statistics import FRAME_TOLERANCE_RATIO, find_frame_outliers
from pdf2fb2.segments.label_roles import HANDWRITTEN_LABEL
from pdf2fb2.segments.position_statistics import OUTLIER_SIGMA_LIMIT, PositionedSegment, find_position_outliers

OUTLIER_DIRECTORY = "outliers"
REGION_PADDING = 8  # pixels kept around the segment box, so strokes touching its edge stay visible


def isolated_region(positioned_segment: PositionedSegment) -> IsolatedRegion:
    """Describe an outlying segment as a padded page region to inspect alone.

    Args:
        positioned_segment: Outlier with its page record.

    Returns:
        Region of the segment's page; the box is padded by ``REGION_PADDING`` (clamped later to the page).
    """
    page_record = positioned_segment.page_record
    left_edge, top_edge, right_edge, bottom_edge = positioned_segment.page_segment["bbox"]
    page_width, page_height = page_record["size"]
    return IsolatedRegion(Path(page_record["image"]), (int(page_width), int(page_height)),
                          [left_edge - REGION_PADDING, top_edge - REGION_PADDING,
                           right_edge + REGION_PADDING, bottom_edge + REGION_PADDING])


def mark_as_handwritten(page_segment: dict) -> None:
    """Relabel a segment as a human-written note, keeping the layout model's label.

    Args:
        page_segment: Segment (mutated: ``label`` set to ``handwritten``, ``original_label`` added).

    Returns:
        None.
    """
    page_segment["original_label"] = page_segment["label"]
    page_segment["label"] = HANDWRITTEN_LABEL


class HandwritingCheck:
    """Marks segments outside the printed frame, and misplaced segments the reader-mark model finds handwriting in.

    Usage:
        Built by the command line with the marks stage; injected into the pipeline and run before the
        whole-document reader-mark cleaning.
    """

    def __init__(self, region_inspector: RegionMarkInspector, sigma_limit: float = OUTLIER_SIGMA_LIMIT,
                 frame_tolerance: float = FRAME_TOLERANCE_RATIO) -> None:
        """Store the inspector and the outlier thresholds.

        Args:
            region_inspector: Reader-mark detection on single page regions.
            sigma_limit: Outlier interval half-width in standard deviations (> 0, default 2.0).
            frame_tolerance: Allowed protrusion past the printed frame, fraction of the page height (>= 0).
        """
        self._region_inspector = region_inspector
        self._sigma_limit = sigma_limit
        self._frame_tolerance = frame_tolerance

    def mark_handwritten_segments(self, page_records: list[dict], work_directory: Path,
                                  stage_progress: StageProgress) -> int:
        """Relabel frame outliers, inspect position outliers for reader marks and relabel the marked ones.

        Args:
            page_records: Segmented page records of the first layout pass (mutated: labels rewritten).
            work_directory: Conversion output directory.
            stage_progress: Receives the check's progress.

        Returns:
            Number of segments marked as handwritten.

        Raises:
            WorkerFailedError: If the reader-mark worker fails.
        """
        frame_outliers = find_frame_outliers(page_records, self._frame_tolerance)
        for frame_outlier in frame_outliers:
            mark_as_handwritten(frame_outlier.page_segment)
        return len(frame_outliers) + self._inspect_position_outliers(page_records, work_directory, stage_progress)

    def _inspect_position_outliers(self, page_records: list[dict], work_directory: Path,
                                   stage_progress: StageProgress) -> int:
        """Inspect the horizontal position outliers for reader marks and relabel the marked ones.

        Args:
            page_records: Segmented page records (mutated: labels rewritten); handwritten segments are not compared.
            work_directory: Conversion output directory.
            stage_progress: Receives the check's progress.

        Returns:
            Number of position outliers marked as handwritten.

        Raises:
            WorkerFailedError: If the reader-mark worker fails.
        """
        outlier_segments = find_position_outliers(page_records, self._sigma_limit)
        if not outlier_segments:
            stage_progress.advance_stage(1, 1, "no outlying segments")
            return 0
        marked_indexes = self._region_inspector.find_marked_regions(
            [isolated_region(outlier_segment) for outlier_segment in outlier_segments],
            work_directory / OUTLIER_DIRECTORY, stage_progress)
        for marked_index in sorted(marked_indexes):
            mark_as_handwritten(outlier_segments[marked_index].page_segment)
        stage_progress.advance_stage(1, 1, f"{len(marked_indexes)} of {len(outlier_segments)} outlying "
                                           "segments are handwritten")
        return len(marked_indexes)
