"""Runs the full book (PDF, DjVu) -> FB2 conversion over injected model stages."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pdf2fb2.document_sources.pdf_source import PdfPassThrough
from pdf2fb2.document_sources.source_formats import DocumentPreparer
from pdf2fb2.fb2_output.fb2_document import BookOptions
from pdf2fb2.pipeline.book_assembly import AssemblyProgress, AssemblyRequest, BookAssembler
from pdf2fb2.pipeline.conversion_reporting import ConversionOutcome, draw_overlays, report_outcome
from pdf2fb2.pipeline.quality_checks import check_segment_boxes
from pdf2fb2.pipeline.stage_contracts import PageSegmenter, SegmentRecognizer
from pdf2fb2.pipeline.stage_planning import OptionalStages, active_stage_names, rescan_trackers
from pdf2fb2.pipeline.workload_counting import count_inspected_tokens, count_text_pages, count_text_tokens
from pdf2fb2.progress.pipeline_progress import PipelineProgress
from pdf2fb2.progress.progress_contracts import ProgressDisplay, RunTimer
from pdf2fb2.progress.stage_names import (
    ASSEMBLING_STAGE,
    DRAWING_STAGE,
    HANDWRITING_STAGE,
    PICTURE_STAGE,
    PROOFREADING_STAGE,
    RECOGNITION_STAGE,
    RENDERING_STAGE,
    RESIDUAL_STAGE,
    REVIEW_STAGE,
    SEGMENT_REVIEW_STAGE,
    SEGMENTATION_STAGE,
    STYLING_STAGE,
    VERIFICATION_STAGE,
)
from pdf2fb2.rendering.pdf_rendering import count_pdf_pages, render_pdf_pages
from pdf2fb2.text_styling.style_markup import apply_word_styles
from pdf2fb2.timekeeping.run_timing import RunTiming
from pdf2fb2.timekeeping.speed_metrics_store import TransientSpeedMetricsStore
from pdf2fb2.timekeeping.time_estimation import ConversionPlan
from pdf2fb2.timekeeping.workload_units import FULL_SCOPE

PAGES_DIRECTORY = "pages"
RunTimerFactory = Callable[[ConversionPlan], RunTimer]

__all__ = ["ConversionPipeline", "ConversionRequest", "OptionalStages", "TimingSetup", "active_stage_names"]


@dataclass(frozen=True)
class ConversionRequest:
    """One conversion to perform.

    Attributes:
        book_path: Input book (PDF or any format of the injected source registry).
        output_directory: Directory for pages, crops, caches, drawings and the FB2.
        book_options: FB2 settings (includes the render DPI).
        draw_segments: Whether to save per-page segmentation drawings.
    """

    book_path: Path
    output_directory: Path
    book_options: BookOptions
    draw_segments: bool


def transient_timer(conversion_plan: ConversionPlan) -> RunTimer:
    """Run timer with the built-in speeds that stores nothing.

    Args:
        conversion_plan: Plan of the conversion.

    Returns:
        The timer.
    """
    return RunTiming(conversion_plan, TransientSpeedMetricsStore())


@dataclass(frozen=True)
class TimingSetup:
    """How a conversion is timed.

    Attributes:
        proofread_scope: ``full`` or ``flagged`` (proofreading speeds are stored per scope).
        timer_factory: Builds the run timer of one conversion from its plan.
    """

    proofread_scope: str = FULL_SCOPE
    timer_factory: RunTimerFactory = transient_timer


class ConversionPipeline:
    """Render -> segment -> [handwriting check] -> [clean marks + re-segment] -> [picture recovery] -> [residual text]
    -> OCR -> [layout review] -> [word boxes] -> [verification + re-reads] -> [styles] -> [proofreading] -> FB2.

    Usage:
        Build with injected stages and display (see ``command_line``) and call ``convert_document``.
    """

    def __init__(self, page_segmenter: PageSegmenter, segment_recognizer: SegmentRecognizer,
                 progress_display: ProgressDisplay, optional_stages: OptionalStages | None = None,
                 document_preparer: DocumentPreparer | None = None, timing_setup: TimingSetup | None = None) -> None:
        """Store the stages.

        Args:
            page_segmenter: Layout stage.
            segment_recognizer: OCR stage.
            progress_display: Shows the overall progress and the current step.
            optional_stages: Stages that may be skipped; None runs none of them.
            document_preparer: Turns the input book into a PDF (None: PDF input only).
            timing_setup: Run timer and proofreading scope (None: built-in speeds, nothing stored).
        """
        self._timing_setup = timing_setup or TimingSetup()
        self._document_preparer = document_preparer or PdfPassThrough()
        self._optional_stages = optional_stages or OptionalStages()
        self._book_assembler = BookAssembler(self._optional_stages.flow_checker)
        self._proofread_text = self._optional_stages.flow_checker is not None
        self._page_segmenter = page_segmenter
        self._segment_recognizer = segment_recognizer
        self._progress_display = progress_display

    def conversion_plan(self, draw_segments: bool) -> ConversionPlan:
        """Stages this pipeline runs, for time estimates.

        Args:
            draw_segments: Whether segmentation drawings are produced.

        Returns:
            Active stages in order and the proofreading scope.
        """
        return ConversionPlan(tuple(active_stage_names(draw_segments, self._optional_stages)),
                              self._timing_setup.proofread_scope)

    def convert_document(self, conversion_request: ConversionRequest) -> int:
        """Convert one book to FB2.

        Args:
            conversion_request: What to convert and where.

        Returns:
            Exit code: 0 when all checks passed, 1 otherwise.

        Raises:
            WorkerFailedError: If a model worker fails.
            FileNotFoundError: If a worker venv is missing.
            DocumentConversionError: If the book cannot be turned into a PDF.
        """
        conversion_plan = self.conversion_plan(conversion_request.draw_segments)
        pipeline_progress = PipelineProgress(self._progress_display, list(conversion_plan.stage_names),
                                             self._timing_setup.timer_factory(conversion_plan))
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
        output_directory.mkdir(parents=True, exist_ok=True)
        pipeline_progress.stage_tracker(RENDERING_STAGE).advance_stage(
            0, 1, f"preparing {conversion_request.book_path.name}")
        pdf_path = self._document_preparer.prepare_document(conversion_request.book_path, output_directory)
        pipeline_progress.revise_workload(page_count=count_pdf_pages(pdf_path))
        page_records = self._read_document(conversion_request, pdf_path, pipeline_progress)
        detected_problems = check_segment_boxes(page_records)
        drawings_directory = None
        if conversion_request.draw_segments:
            drawings_directory = draw_overlays(page_records, output_directory,
                                               pipeline_progress.stage_tracker(DRAWING_STAGE))
        fb2_path, book_statistics, validation_problems = self._book_assembler.assemble_book(
            AssemblyRequest(pdf_path, output_directory, conversion_request.book_options),
            page_records, AssemblyProgress(
                pipeline_progress.stage_tracker(ASSEMBLING_STAGE),
                pipeline_progress.stage_tracker(PROOFREADING_STAGE) if self._proofread_text else None))
        if self._proofread_text:
            pipeline_progress.revise_workload(inspected_token_count=count_inspected_tokens(
                book_statistics, count_text_tokens(page_records)))
        return ConversionOutcome(fb2_path, book_statistics, detected_problems + validation_problems,
                                 drawings_directory)

    def _read_document(self, conversion_request: ConversionRequest, pdf_path: Path,
                       pipeline_progress: PipelineProgress) -> list[dict]:
        """Segment and correct the segmentation, OCR, review the labels, verify the text and apply word styles.

        Args:
            conversion_request: What to convert and where.
            pdf_path: PDF with the book's pages.
            pipeline_progress: Overall progress shared by the stages.

        Returns:
            Final OCR'd (and styled) page records.
        """
        output_directory = conversion_request.output_directory
        page_records = self._correct_segmentation(self._segment_document(conversion_request, pdf_path,
                                                                         pipeline_progress),
                                                  output_directory, pipeline_progress)
        pipeline_progress.revise_workload(text_page_count=count_text_pages(page_records))
        self._segment_recognizer.recognize_segments(page_records, output_directory,
                                                    pipeline_progress.stage_tracker(RECOGNITION_STAGE))
        pipeline_progress.revise_workload(token_count=count_text_tokens(page_records))
        optional_stages = self._optional_stages
        if optional_stages.layout_reviewer is not None:
            optional_stages.layout_reviewer.review_layout(page_records, pipeline_progress.stage_tracker(REVIEW_STAGE))
        if optional_stages.style_detector is not None:
            optional_stages.style_detector.detect_text_styles(page_records, output_directory,
                                                              pipeline_progress.stage_tracker(STYLING_STAGE))
        if optional_stages.text_verifier is not None:
            optional_stages.text_verifier.verify_segments(page_records, output_directory,
                                                          pipeline_progress.stage_tracker(VERIFICATION_STAGE))
        if optional_stages.style_detector is not None:
            apply_word_styles(page_records)
        return page_records

    def _correct_segmentation(self, page_records: list[dict], output_directory: Path,
                              pipeline_progress: PipelineProgress) -> list[dict]:
        """Mark handwriting, clean + re-segment marked pages, recover pictures, add residual text, review labels.

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
        if optional_stages.segment_reviewer is not None:
            optional_stages.segment_reviewer.review_layout(page_records,
                                                           pipeline_progress.stage_tracker(SEGMENT_REVIEW_STAGE))
        return page_records

    def _segment_document(self, conversion_request: ConversionRequest, pdf_path: Path,
                          pipeline_progress: PipelineProgress) -> list[dict]:
        """Render the PDF pages and segment them.

        Args:
            conversion_request: What to convert and where.
            pdf_path: PDF with the book's pages.
            pipeline_progress: Overall progress shared by the stages.

        Returns:
            Segmented page records.
        """
        output_directory = conversion_request.output_directory
        rendering_progress = pipeline_progress.stage_tracker(RENDERING_STAGE)
        page_images = render_pdf_pages(
            pdf_path, output_directory / PAGES_DIRECTORY,
            conversion_request.book_options.segmentation_dpi,
            lambda rendered_count, page_count: rendering_progress.advance_stage(
                rendered_count, page_count, f"extracting pages from pdf {rendered_count}/{page_count}"))
        return self._page_segmenter.segment_pages(page_images, output_directory,
                                                  pipeline_progress.stage_tracker(SEGMENTATION_STAGE))
