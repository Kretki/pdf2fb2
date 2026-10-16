"""Runs the full PDF -> FB2 conversion over injected model stages."""

import json
from dataclasses import dataclass
from pathlib import Path

from pdf2fb2.fb2_output.fb2_document import BookOptions, build_fb2_document
from pdf2fb2.pipeline.quality_checks import check_segment_boxes, validate_fb2_file
from pdf2fb2.pipeline.stage_contracts import MarkedPageRescanner, PageSegmenter, RescanProgress, SegmentRecognizer
from pdf2fb2.progress.pipeline_progress import (
    ASSEMBLING_STAGE,
    CLEANING_STAGE,
    DRAWING_STAGE,
    RECOGNITION_STAGE,
    RENDERING_STAGE,
    RESCAN_RECOGNITION_STAGE,
    RESCAN_SEGMENTATION_STAGE,
    SEGMENTATION_STAGE,
    PipelineProgress,
)
from pdf2fb2.progress.progress_contracts import ProgressDisplay, StageProgress
from pdf2fb2.rendering.pdf_rendering import render_pdf_pages
from pdf2fb2.rendering.segment_drawing import draw_segment_overlay

PAGES_DIRECTORY = "pages"
VISUALIZATION_DIRECTORY = "vis"
SEGMENTS_FILE = "segments.json"
ASSEMBLING_STEPS = 3


@dataclass(frozen=True)
class ConversionRequest:
    """One conversion to perform.

    Attributes:
        pdf_path: Input PDF.
        output_directory: Directory for pages, crops, caches, drawings and the FB2.
        book_options: FB2 settings (includes the render DPI).
        draw_segments: Whether to save per-page segmentation drawings.
    """

    pdf_path: Path
    output_directory: Path
    book_options: BookOptions
    draw_segments: bool


@dataclass(frozen=True)
class ConversionOutcome:
    """What a finished conversion produced.

    Attributes:
        fb2_path: Written FB2 file.
        book_statistics: Element counts of the FB2.
        detected_problems: Problems found by the segmentation and FB2 checks.
        drawings_directory: Segmentation drawings directory, or None when disabled.
    """

    fb2_path: Path
    book_statistics: dict
    detected_problems: list[str]
    drawings_directory: Path | None


def active_stage_names(draw_segments: bool, clean_marks: bool = False) -> list[str]:
    """List the progress stages of a conversion in execution order.

    Args:
        draw_segments: Whether segmentation drawings are produced.
        clean_marks: Whether reader marks are removed (cleaning + rescan of marked pages after OCR).

    Returns:
        Stage names for ``PipelineProgress``.
    """
    rescan_stages = [CLEANING_STAGE, RESCAN_SEGMENTATION_STAGE, RESCAN_RECOGNITION_STAGE] if clean_marks else []
    optional_stages = [DRAWING_STAGE] if draw_segments else []
    return [RENDERING_STAGE, SEGMENTATION_STAGE, RECOGNITION_STAGE, *rescan_stages, *optional_stages,
            ASSEMBLING_STAGE]


def rescan_trackers(pipeline_progress: PipelineProgress) -> RescanProgress:
    """Progress sinks of the reader-mark rescan.

    Args:
        pipeline_progress: Overall progress.

    Returns:
        Trackers of the cleaning, re-segmentation and re-recognition stages.
    """
    return RescanProgress(pipeline_progress.stage_tracker(CLEANING_STAGE),
                          pipeline_progress.stage_tracker(RESCAN_SEGMENTATION_STAGE),
                          pipeline_progress.stage_tracker(RESCAN_RECOGNITION_STAGE))


def draw_overlays(page_records: list[dict], output_directory: Path, stage_progress: StageProgress) -> Path:
    """Save segmentation drawings for every page.

    Args:
        page_records: Segmented pages.
        output_directory: Conversion output directory.
        stage_progress: Receives ``drawing segments of page k/n`` updates.

    Returns:
        Directory containing the drawings.
    """
    drawings_directory = output_directory / VISUALIZATION_DIRECTORY
    drawings_directory.mkdir(exist_ok=True)
    page_count = len(page_records)
    for page_position, page_record in enumerate(page_records, start=1):
        stage_progress.advance_stage(page_position - 1, page_count,
                                     f"drawing segments of page {page_position}/{page_count}")
        draw_segment_overlay(Path(page_record["image"]), page_record["segments"],
                             drawings_directory / f"page_{page_record['page']:04d}_segments.png")
    return drawings_directory


def assemble_book(conversion_request: ConversionRequest, page_records: list[dict],
                  stage_progress: StageProgress) -> tuple[Path, dict, list[str]]:
    """Save the segments, build and write the FB2, then validate it.

    Args:
        conversion_request: Conversion settings.
        page_records: OCR'd page records.
        stage_progress: Receives one update per step.

    Returns:
        ``(fb2_path, statistics, validation_problems)``.
    """
    output_directory = conversion_request.output_directory
    stage_progress.advance_stage(0, ASSEMBLING_STEPS, f"saving {SEGMENTS_FILE}")
    (output_directory / SEGMENTS_FILE).write_text(json.dumps(page_records, ensure_ascii=False, indent=1),
                                                  encoding="utf-8")
    stage_progress.advance_stage(1, ASSEMBLING_STEPS, "building FB2 document")
    fb2_markup, book_statistics = build_fb2_document(conversion_request.pdf_path, page_records,
                                                  conversion_request.book_options)
    fb2_path = output_directory / f"{conversion_request.pdf_path.stem}.fb2"
    fb2_path.write_text(fb2_markup, encoding="utf-8")
    stage_progress.advance_stage(2, ASSEMBLING_STEPS, "validating FB2")
    validation_problems = validate_fb2_file(fb2_path)
    stage_progress.advance_stage(ASSEMBLING_STEPS, ASSEMBLING_STEPS, "FB2 written")
    return fb2_path, book_statistics, validation_problems


def report_outcome(conversion_outcome: ConversionOutcome) -> int:
    """Print the conversion summary.

    Args:
        conversion_outcome: Result of the conversion.

    Returns:
        Exit code: 0 when all checks passed, 1 otherwise.
    """
    print(f"FB2: {conversion_outcome.fb2_path}  {conversion_outcome.book_statistics}")
    if conversion_outcome.drawings_directory is not None:
        print(f"Segment drawings: {conversion_outcome.drawings_directory}")
    if not conversion_outcome.detected_problems:
        print("All checks passed.")
        return 0
    print("\nCHECKS FAILED:")
    for detected_problem in conversion_outcome.detected_problems:
        print("  -", detected_problem)
    return 1


class ConversionPipeline:
    """Render -> segment -> OCR -> [clean marks + rescan] -> FB2 -> checks, with one overall progress display.

    Usage:
        Build with injected stages and display (see ``command_line``) and call ``convert_document``.
    """

    def __init__(self, page_segmenter: PageSegmenter, segment_recognizer: SegmentRecognizer,
                 progress_display: ProgressDisplay, mark_rescanner: MarkedPageRescanner | None = None) -> None:
        """Store the stages.

        Args:
            page_segmenter: Layout stage.
            segment_recognizer: OCR stage.
            progress_display: Shows the overall progress and the current step.
            mark_rescanner: Cleans pages with reader marks and rescans them after OCR, or None to skip.
        """
        self._mark_rescanner = mark_rescanner
        self._page_segmenter = page_segmenter
        self._segment_recognizer = segment_recognizer
        self._progress_display = progress_display

    def convert_document(self, conversion_request: ConversionRequest) -> int:
        """Convert one PDF to FB2.

        Args:
            conversion_request: What to convert and where.

        Returns:
            Exit code: 0 when all checks passed, 1 otherwise.

        Raises:
            WorkerFailedError: If a model worker fails.
            FileNotFoundError: If a worker venv is missing.
        """
        pipeline_progress = PipelineProgress(self._progress_display,
                                             active_stage_names(conversion_request.draw_segments,
                                                                self._mark_rescanner is not None))
        conversion_finished = False
        try:
            conversion_outcome = self._run_stages(conversion_request, pipeline_progress)
            conversion_finished = True
        finally:
            pipeline_progress.close_progress(conversion_finished)
        return report_outcome(conversion_outcome)

    def _run_stages(self, conversion_request: ConversionRequest,
                    pipeline_progress: PipelineProgress) -> ConversionOutcome:
        """Run every stage in order.

        Args:
            conversion_request: What to convert and where.
            pipeline_progress: Overall progress shared by the stages.

        Returns:
            What the conversion produced.
        """
        output_directory = conversion_request.output_directory
        page_records = self._segment_document(conversion_request, pipeline_progress)
        self._segment_recognizer.recognize_segments(page_records, output_directory,
                                                    pipeline_progress.stage_tracker(RECOGNITION_STAGE))
        if self._mark_rescanner is not None:
            page_records = self._mark_rescanner.rescan_marked_pages(page_records, output_directory,
                                                                    rescan_trackers(pipeline_progress))
        detected_problems = check_segment_boxes(page_records)
        drawings_directory = None
        if conversion_request.draw_segments:
            drawings_directory = draw_overlays(page_records, output_directory,
                                               pipeline_progress.stage_tracker(DRAWING_STAGE))
        fb2_path, book_statistics, validation_problems = assemble_book(
            conversion_request, page_records, pipeline_progress.stage_tracker(ASSEMBLING_STAGE))
        return ConversionOutcome(fb2_path, book_statistics, detected_problems + validation_problems,
                                 drawings_directory)

    def _segment_document(self, conversion_request: ConversionRequest,
                          pipeline_progress: PipelineProgress) -> list[dict]:
        """Render the PDF pages and segment them.

        Args:
            conversion_request: What to convert and where.
            pipeline_progress: Overall progress shared by the stages.

        Returns:
            Segmented page records.
        """
        output_directory = conversion_request.output_directory
        output_directory.mkdir(parents=True, exist_ok=True)
        rendering_progress = pipeline_progress.stage_tracker(RENDERING_STAGE)
        page_images = render_pdf_pages(
            conversion_request.pdf_path, output_directory / PAGES_DIRECTORY,
            conversion_request.book_options.segmentation_dpi,
            lambda page_number, page_count: rendering_progress.advance_stage(
                page_number - 1, page_count, f"extracting pages from pdf {page_number}/{page_count}"))
        return self._page_segmenter.segment_pages(page_images, output_directory,
                                                  pipeline_progress.stage_tracker(SEGMENTATION_STAGE))
