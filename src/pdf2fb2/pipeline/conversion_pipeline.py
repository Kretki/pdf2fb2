"""Runs the full PDF -> FB2 conversion over injected model stages."""

from dataclasses import dataclass
from pathlib import Path

from pdf2fb2.fb2_output.fb2_document import BookOptions
from pdf2fb2.pipeline.book_assembly import AssemblyProgress, AssemblyRequest, BookAssembler
from pdf2fb2.pipeline.quality_checks import check_segment_boxes
from pdf2fb2.pipeline.stage_contracts import (
    FlowChecker,
    HandwritingDetector,
    MarkedPageRescanner,
    PageSegmenter,
    PictureRecoverer,
    RescanProgress,
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


def active_stage_names(draw_segments: bool, clean_marks: bool = False, detect_styles: bool = False,
                       proofread_text: bool = False, check_handwriting: bool = False,
                       recover_pictures: bool = False) -> list[str]:
    """List the progress stages of a conversion in execution order.

    Args:
        draw_segments: Whether segmentation drawings are produced.
        clean_marks: Whether reader marks are removed (cleaning + re-segmentation of marked pages before OCR).
        detect_styles: Whether word styles (bold/italic/strikeout) are detected after OCR.
        proofread_text: Whether the book text is proofread with the language model before rendering.
        check_handwriting: Whether position-outlier segments are checked for handwriting before cleaning.
        recover_pictures: Whether off-size lettering is replaced by pictures before OCR.

    Returns:
        Stage names for ``PipelineProgress``.
    """
    handwriting_stages = [HANDWRITING_STAGE] if check_handwriting else []
    rescan_stages = [CLEANING_STAGE, RESCAN_SEGMENTATION_STAGE] if clean_marks else []
    picture_stages = [PICTURE_STAGE] if recover_pictures else []
    styling_stages = [STYLING_STAGE] if detect_styles else []
    optional_stages = [DRAWING_STAGE] if draw_segments else []
    proofreading_stages = [PROOFREADING_STAGE] if proofread_text else []
    return [RENDERING_STAGE, SEGMENTATION_STAGE, *handwriting_stages, *rescan_stages, *picture_stages,
            RECOGNITION_STAGE,
            *styling_stages, *optional_stages, *proofreading_stages, ASSEMBLING_STAGE]


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
    """Render -> segment -> [handwriting check] -> [clean marks + re-segment] -> [picture recovery] -> OCR -> [styles]
    -> [proofreading] -> FB2.

    Usage:
        Build with injected stages and display (see ``command_line``) and call ``convert_document``.
    """

    def __init__(self, page_segmenter: PageSegmenter, segment_recognizer: SegmentRecognizer,
                 progress_display: ProgressDisplay, mark_rescanner: MarkedPageRescanner | None = None,
                 style_detector: TextStyleDetector | None = None, flow_checker: FlowChecker | None = None,
                 handwriting_detector: HandwritingDetector | None = None,
                 picture_recoverer: PictureRecoverer | None = None) -> None:
        """Store the stages.

        Args:
            page_segmenter: Layout stage.
            segment_recognizer: OCR stage.
            progress_display: Shows the overall progress and the current step.
            mark_rescanner: Cleans pages with reader marks and re-segments them before OCR, or None to skip.
            style_detector: Detects word bold/italic/strikeout in text segments, or None to skip.
            flow_checker: Proofreads and rearranges the book flow with a language model, or None to skip.
            handwriting_detector: Marks position-outlier segments with handwriting before mark cleaning,
                or None to skip.
            picture_recoverer: Replaces lettering that does not fit the book's font size by pictures before OCR,
                or None to skip.
        """
        self._handwriting_detector = handwriting_detector
        self._picture_recoverer = picture_recoverer
        self._book_assembler = BookAssembler(flow_checker)
        self._proofread_text = flow_checker is not None
        self._mark_rescanner = mark_rescanner
        self._style_detector = style_detector
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
                                                                self._mark_rescanner is not None,
                                                                self._style_detector is not None,
                                                                self._proofread_text,
                                                                self._handwriting_detector is not None,
                                                                self._picture_recoverer is not None))
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
        """Segment, mark handwritten segments, clean + re-segment marked pages, recover pictures, OCR, apply styles.

        Args:
            conversion_request: What to convert and where.
            pipeline_progress: Overall progress shared by the stages.

        Returns:
            Final OCR'd (and styled) page records.
        """
        output_directory = conversion_request.output_directory
        page_records = self._segment_document(conversion_request, pipeline_progress)
        if self._handwriting_detector is not None:
            self._handwriting_detector.mark_handwritten_segments(page_records, output_directory,
                                                                 pipeline_progress.stage_tracker(HANDWRITING_STAGE))
        if self._mark_rescanner is not None:
            page_records = self._mark_rescanner.rescan_marked_pages(page_records, output_directory,
                                                                    rescan_trackers(pipeline_progress))
        if self._picture_recoverer is not None:
            self._picture_recoverer.recover_pictures(page_records, output_directory,
                                                     pipeline_progress.stage_tracker(PICTURE_STAGE))
        self._segment_recognizer.recognize_segments(page_records, output_directory,
                                                    pipeline_progress.stage_tracker(RECOGNITION_STAGE))
        if self._style_detector is not None:
            self._style_detector.detect_text_styles(page_records, output_directory,
                                                    pipeline_progress.stage_tracker(STYLING_STAGE))
            apply_word_styles(page_records)
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
