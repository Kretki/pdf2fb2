"""Wiring of the conversion pipeline: builds every stage from the parsed command line.

Usage:
    ``build_pipeline(parsed_arguments, progress_display)`` is called by ``command_line.run_command_line``.
"""

import argparse

from pdf2fb2.document_sources.djvu_source import DjvuToPdfConverter
from pdf2fb2.document_sources.pdf_source import PdfPassThrough
from pdf2fb2.document_sources.source_formats import SourceFormatRegistry
from pdf2fb2.model_stages.layout_stage import LayoutStage
from pdf2fb2.model_stages.marks_stage import MarksStage
from pdf2fb2.model_stages.ocr_stage import OcrBackendSettings, OcrStage
from pdf2fb2.model_stages.proofread_stage import ProofreadStage
from pdf2fb2.model_stages.style_stage import StyleStage
from pdf2fb2.pipeline.conversion_pipeline import ConversionPipeline, OptionalStages
from pdf2fb2.pipeline.handwriting_check import HandwritingCheck
from pdf2fb2.pipeline.layout_review import SEGMENT_REVIEW_RULES, LayoutReview
from pdf2fb2.pipeline.mark_rescan import MarkRescanner
from pdf2fb2.pipeline.picture_recovery import PictureRecovery
from pdf2fb2.pipeline.residual_rescan import ResidualTextRescan
from pdf2fb2.progress.progress_contracts import ProgressDisplay
from pdf2fb2.project_paths import READERMARKS_CHECKPOINT_DIRECTORY
from pdf2fb2.proofreading.flow_proofreader import LanguageModelProofreader
from pdf2fb2.segments.glyph_measurement import OpenCvGlyphMeter
from pdf2fb2.worker_runtime.worker_environment import (
    LAYOUT_WORKER,
    MARKS_WORKER,
    OCR_WORKER,
    PROOFREAD_WORKER,
    STYLE_WORKER,
)
from pdf2fb2.worker_runtime.worker_launcher import SubprocessWorkerLauncher
from worker_protocol.ocr_job_schema import HTTP_BACKEND, TRANSFORMERS_BACKEND


def build_source_registry() -> SourceFormatRegistry:
    """Wire the supported input formats.

    Returns:
        Registry for ``.pdf`` (used as is) and ``.djvu``/``.djv`` (converted with ``ddjvu``).
    """
    djvu_converter = DjvuToPdfConverter()
    return SourceFormatRegistry({".pdf": PdfPassThrough(), ".djvu": djvu_converter, ".djv": djvu_converter})


def build_ocr_settings(parsed_arguments: argparse.Namespace) -> OcrBackendSettings:
    """Choose the OCR backend: HTTP server when ``--qianfan-url`` is given, else local weights.

    Args:
        parsed_arguments: Parsed command line.

    Returns:
        Settings for the OCR stage.
    """
    use_server = parsed_arguments.qianfan_url is not None
    return OcrBackendSettings(
        backend_name=HTTP_BACKEND if use_server else TRANSFORMERS_BACKEND,
        model_directory=None if use_server else parsed_arguments.qianfan_directory,
        device_name=parsed_arguments.device_name,
        server_url=parsed_arguments.qianfan_url,
        server_model=parsed_arguments.qianfan_model if use_server else None,
        parallel_requests=parsed_arguments.parallel_requests,
    )


def build_picture_recovery(parsed_arguments: argparse.Namespace,
                           worker_launcher: SubprocessWorkerLauncher) -> PictureRecovery | None:
    """Wire the picture recovery with the OpenCV glyph meter and a low-threshold layout stage.

    Args:
        parsed_arguments: Parsed command line.
        worker_launcher: Starts the layout worker.

    Returns:
        The picture recovery, or None with ``--no-picture-recovery``.
    """
    if parsed_arguments.skip_picture_recovery:
        return None
    proposal_stage = LayoutStage(worker_launcher, LAYOUT_WORKER, parsed_arguments.layout_model_directory,
                                 parsed_arguments.device_name, parsed_arguments.picture_threshold)
    return PictureRecovery(OpenCvGlyphMeter(), proposal_stage)


def build_residual_rescan(parsed_arguments: argparse.Namespace,
                          worker_launcher: SubprocessWorkerLauncher) -> ResidualTextRescan | None:
    """Wire the residual-text rescan with the OpenCV glyph meter and a layout stage of its own threshold.

    Args:
        parsed_arguments: Parsed command line.
        worker_launcher: Starts the layout worker.

    Returns:
        The rescan, or None with ``--no-residual-text``.
    """
    if parsed_arguments.skip_residual_text:
        return None
    proposal_stage = LayoutStage(worker_launcher, LAYOUT_WORKER, parsed_arguments.layout_model_directory,
                                 parsed_arguments.device_name, parsed_arguments.residual_threshold)
    return ResidualTextRescan(OpenCvGlyphMeter(), proposal_stage)


def build_pipeline(parsed_arguments: argparse.Namespace, progress_display: ProgressDisplay) -> ConversionPipeline:
    """Wire the stages with the subprocess launcher, the worker venvs and the progress display.

    Args:
        parsed_arguments: Parsed command line.
        progress_display: Shows the overall progress bar and the current step.

    Returns:
        Ready-to-run pipeline.
    """
    worker_launcher = SubprocessWorkerLauncher()
    layout_stage = LayoutStage(worker_launcher, LAYOUT_WORKER, parsed_arguments.layout_model_directory,
                               parsed_arguments.device_name, parsed_arguments.score_threshold)
    ocr_stage = OcrStage(worker_launcher, OCR_WORKER, build_ocr_settings(parsed_arguments))
    mark_rescanner = None
    handwriting_check = None
    if not parsed_arguments.keep_marks:
        marks_stage = MarksStage(worker_launcher, MARKS_WORKER,
                                 parsed_arguments.marks_model_directory or READERMARKS_CHECKPOINT_DIRECTORY,
                                 parsed_arguments.device_name, parsed_arguments.debug_marks)
        mark_rescanner = MarkRescanner(marks_stage, layout_stage)
        handwriting_check = HandwritingCheck(marks_stage)
    style_stage = None if parsed_arguments.skip_styles else StyleStage(
        worker_launcher, STYLE_WORKER, parsed_arguments.textar_model_directory,
        parsed_arguments.doctr_model_directory, parsed_arguments.device_name)
    flow_checker = None if parsed_arguments.skip_proofread else LanguageModelProofreader(ProofreadStage(
        worker_launcher, PROOFREAD_WORKER, parsed_arguments.proofread_model_directory, parsed_arguments.device_name))
    return ConversionPipeline(layout_stage, ocr_stage, progress_display, OptionalStages(
        handwriting_detector=handwriting_check, mark_rescanner=mark_rescanner,
        picture_recoverer=build_picture_recovery(parsed_arguments, worker_launcher),
        residual_finder=build_residual_rescan(parsed_arguments, worker_launcher),
        segment_reviewer=None if parsed_arguments.skip_layout_review else LayoutReview(SEGMENT_REVIEW_RULES),
        layout_reviewer=None if parsed_arguments.skip_layout_review else LayoutReview(),
        style_detector=style_stage, flow_checker=flow_checker), build_source_registry())
