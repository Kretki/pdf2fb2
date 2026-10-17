"""Runs the full PDF -> FB2 conversion over injected model stages."""

from dataclasses import dataclass
from pathlib import Path

from pdf2fb2.fb2_output.fb2_document import BookOptions
from pdf2fb2.pipeline.book_assembly import AssemblyProgress, AssemblyRequest, BookAssembler
from pdf2fb2.pipeline.quality_checks import check_segment_boxes
from pdf2fb2.pipeline.stage_contracts import (
    FlowChecker,
    HandwritingDetector,
    LayoutReviewer,
    MarkedPageRescanner,
    PageSegmenter,
    PictureRecoverer,
    RescanProgress,
    ResidualTextFinder,
    SegmentRecognizer,
    TextStyleDetector,
)
from pdf2fb2.progress.pipeline_progress import (
    ASSEMBLING_STAGE,
    CLEANING_STAGE,
    DRAWING_STAGE,
    HANDWRITING_STAGE,
    PICTURE_STAGE,
    PROOFREADING_STAGE,
    RECOGNITION_STAGE,
    RENDERING_STAGE,
    RESCAN_SEGMENTATION_STAGE,
    RESIDUAL_STAGE,
    REVIEW_STAGE,
    SEGMENTATION_STAGE,
    STYLING_STAGE,
    PipelineProgress,
)
from pdf2fb2.progress.progress_contracts import ProgressDisplay, StageProgress
from pdf2fb2.rendering.pdf_rendering import render_pdf_pages
from pdf2fb2.rendering.segment_drawing import draw_segment_overlay
from pdf2fb2.text_styling.style_markup import apply_word_styles

PAGES_DIRECTORY = "pages"
VISUALIZATION_DIRECTORY = "vis"


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


@dataclass(frozen=True)
class OptionalStages:
    """Stages a conversion can run or skip; None skips the stage.

    Attributes:
        handwriting_detector: Marks handwritten segments before mark cleaning.
        mark_rescanner: Cleans pages with reader marks and re-segments them before OCR.
        picture_recoverer: Replaces lettering that does not fit the book's font size by pictures before OCR.
        residual_finder: Adds segments for text left outside every segment, last step before OCR.
        layout_reviewer: Corrects labels (page numbers, headings) once the text is known, before word styles.
        style_detector: Detects word bold/italic/strikeout in text segments.
        flow_checker: Proofreads and rearranges the book flow with a language model.
    """

    handwriting_detector: HandwritingDetector | None = None
    mark_rescanner: MarkedPageRescanner | None = None
    picture_recoverer: PictureRecoverer | None = None
    residual_finder: ResidualTextFinder | None = None
    layout_reviewer: LayoutReviewer | None = None
    style_detector: TextStyleDetector | None = None
    flow_checker: FlowChecker | None = None


def active_stage_names(draw_segments: bool, optional_stages: OptionalStages) -> list[str]:
    """List the progress stages of a conversion in execution order.

    Args:
        draw_segments: Whether segmentation drawings are produced.
        optional_stages: Optional stages of the conversion (None entries are skipped).

    Returns:
        Stage names for ``PipelineProgress``.
    """
    stage_switches = (
        (optional_stages.handwriting_detector is not None, [HANDWRITING_STAGE]),
        (optional_stages.mark_rescanner is not None, [CLEANING_STAGE, RESCAN_SEGMENTATION_STAGE]),
        (optional_stages.picture_recoverer is not None, [PICTURE_STAGE]),
        (optional_stages.residual_finder is not None, [RESIDUAL_STAGE]),
        (True, [RECOGNITION_STAGE]),
        (optional_stages.layout_reviewer is not None, [REVIEW_STAGE]),
        (optional_stages.style_detector is not None, [STYLING_STAGE]),
        (draw_segments, [DRAWING_STAGE]),
        (optional_stages.flow_checker is not None, [PROOFREADING_STAGE]),
    )
    return [RENDERING_STAGE, SEGMENTATION_STAGE,
            *(stage_name for stage_enabled, stage_names in stage_switches if stage_enabled
              for stage_name in stage_names), ASSEMBLING_STAGE]


def rescan_trackers(pipeline_progress: PipelineProgress) -> RescanProgress:
    """Progress sinks of the reader-mark rescan.

    Args:
        pipeline_progress: Overall progress.

    Returns:
        Trackers of the cleaning and re-segmentation stages.
    """
    return RescanProgress(pipeline_progress.stage_tracker(CLEANING_STAGE),
                          pipeline_progress.stage_tracker(RESCAN_SEGMENTATION_STAGE))


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
    """Render -> segment -> [handwriting check] -> [clean marks + re-segment] -> [picture recovery] -> [residual text]
    -> OCR -> [layout review] -> [styles] -> [proofreading] -> FB2.

    Usage:
        Build with injected stages and display (see ``command_line``) and call ``convert_document``.
    """

    def __init__(self, page_segmenter: PageSegmenter, segment_recognizer: SegmentRecognizer,
                 progress_display: ProgressDisplay, optional_stages: OptionalStages | None = None) -> None:
        """Store the stages.

        Args:
            page_segmenter: Layout stage.
            segment_recognizer: OCR stage.
            progress_display: Shows the overall progress and the current step.
            optional_stages: Stages that may be skipped; None runs none of them.
        """
        self._optional_stages = optional_stages or OptionalStages()
        self._book_assembler = BookAssembler(self._optional_stages.flow_checker)
        self._proofread_text = self._optional_stages.flow_checker is not None
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
                                                                self._optional_stages))
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
        page_records = self._read_document(conversion_request, pipeline_progress)
        detected_problems = check_segment_boxes(page_records)
        drawings_directory = None
        if conversion_request.draw_segments:
            drawings_directory = draw_overlays(page_records, output_directory,
                                               pipeline_progress.stage_tracker(DRAWING_STAGE))
        fb2_path, book_statistics, validation_problems = self._book_assembler.assemble_book(
            AssemblyRequest(conversion_request.pdf_path, output_directory, conversion_request.book_options),
            page_records, AssemblyProgress(
                pipeline_progress.stage_tracker(ASSEMBLING_STAGE),
                pipeline_progress.stage_tracker(PROOFREADING_STAGE) if self._proofread_text else None))
        return ConversionOutcome(fb2_path, book_statistics, detected_problems + validation_problems,
                                 drawings_directory)

    def _read_document(self, conversion_request: ConversionRequest,
                       pipeline_progress: PipelineProgress) -> list[dict]:
        """Segment and correct the segmentation, OCR, review the labels and apply word styles.

        Args:
            conversion_request: What to convert and where.
            pipeline_progress: Overall progress shared by the stages.

        Returns:
            Final OCR'd (and styled) page records.
        """
        output_directory = conversion_request.output_directory
        page_records = self._correct_segmentation(self._segment_document(conversion_request, pipeline_progress),
                                                  output_directory, pipeline_progress)
        self._segment_recognizer.recognize_segments(page_records, output_directory,
                                                    pipeline_progress.stage_tracker(RECOGNITION_STAGE))
        optional_stages = self._optional_stages
        if optional_stages.layout_reviewer is not None:
            optional_stages.layout_reviewer.review_layout(page_records, pipeline_progress.stage_tracker(REVIEW_STAGE))
        if optional_stages.style_detector is not None:
            optional_stages.style_detector.detect_text_styles(page_records, output_directory,
                                                              pipeline_progress.stage_tracker(STYLING_STAGE))
            apply_word_styles(page_records)
        return page_records

    def _correct_segmentation(self, page_records: list[dict], output_directory: Path,
                              pipeline_progress: PipelineProgress) -> list[dict]:
        """Mark handwriting, clean + re-segment marked pages, recover pictures and add residual text.

        Args:
            page_records: First-pass segmented page records (mutated).
            output_directory: Conversion output directory.
            pipeline_progress: Overall progress shared by the stages.

        Returns:
            Final segmented page records, ready for OCR.
        """
        optional_stages = self._optional_stages
        if optional_stages.handwriting_detector is not None:
            optional_stages.handwriting_detector.mark_handwritten_segments(
                page_records, output_directory, pipeline_progress.stage_tracker(HANDWRITING_STAGE))
        if optional_stages.mark_rescanner is not None:
            page_records = optional_stages.mark_rescanner.rescan_marked_pages(page_records, output_directory,
                                                                              rescan_trackers(pipeline_progress))
        if optional_stages.picture_recoverer is not None:
            optional_stages.picture_recoverer.recover_pictures(page_records, output_directory,
                                                               pipeline_progress.stage_tracker(PICTURE_STAGE))
        if optional_stages.residual_finder is not None:
            optional_stages.residual_finder.add_residual_text(page_records, output_directory,
                                                              pipeline_progress.stage_tracker(RESIDUAL_STAGE))
        return page_records

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
