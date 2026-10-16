"""Final assembly: save the segments, build the flow, proofread it (optional), write and validate the FB2."""

import json
from dataclasses import dataclass
from pathlib import Path

from pdf2fb2.fb2_output.fb2_document import BookOptions, prepare_book, render_prepared_book
from pdf2fb2.pipeline.quality_checks import validate_fb2_file
from pdf2fb2.pipeline.stage_contracts import FlowChecker
from pdf2fb2.progress.progress_contracts import StageProgress

SEGMENTS_FILE = "segments.json"
ASSEMBLING_STEPS = 3


@dataclass(frozen=True)
class AssemblyRequest:
    """What to assemble.

    Attributes:
        pdf_path: Source PDF.
        output_directory: Conversion output directory.
        book_options: FB2 settings.
    """

    pdf_path: Path
    output_directory: Path
    book_options: BookOptions


@dataclass(frozen=True)
class AssemblyProgress:
    """Progress sinks of the assembly.

    Attributes:
        assembling_progress: Saving, building, writing and validating.
        proofreading_progress: Language-model proofreading (unused without a checker).
    """

    assembling_progress: StageProgress
    proofreading_progress: StageProgress | None


class BookAssembler:
    """Builds the FB2 from OCR'd page records, with an optional proofreading step on the flow.

    Usage:
        Build once (inject a ``FlowChecker`` or None) and call ``assemble_book``.
    """

    def __init__(self, flow_checker: FlowChecker | None) -> None:
        """Store the optional checker.

        Args:
            flow_checker: Proofreads and rearranges the book flow, or None to skip.
        """
        self._flow_checker = flow_checker

    def assemble_book(self, assembly_request: AssemblyRequest, page_records: list[dict],
                      assembly_progress: AssemblyProgress) -> tuple[Path, dict, list[str]]:
        """Save the segments, build (and proofread) the flow, write and validate the FB2.

        Args:
            assembly_request: Paths and options.
            page_records: OCR'd page records.
            assembly_progress: Progress sinks.

        Returns:
            ``(fb2_path, statistics, validation_problems)``; statistics include the proofreading summary.

        Raises:
            WorkerFailedError: If the proofreading worker fails.
        """
        stage_progress = assembly_progress.assembling_progress
        output_directory = assembly_request.output_directory
        stage_progress.advance_stage(0, ASSEMBLING_STEPS, f"saving {SEGMENTS_FILE}")
        (output_directory / SEGMENTS_FILE).write_text(json.dumps(page_records, ensure_ascii=False, indent=1),
                                                      encoding="utf-8")
        stage_progress.advance_stage(1, ASSEMBLING_STEPS, "building FB2 document")
        prepared_book = prepare_book(assembly_request.pdf_path, page_records, assembly_request.book_options)
        proofreading_summary = {}
        if self._flow_checker is not None and assembly_progress.proofreading_progress is not None:
            proofreading_summary = self._flow_checker.proofread_flow(prepared_book.book_flow, output_directory,
                                                                     assembly_progress.proofreading_progress)
        fb2_markup, book_statistics = render_prepared_book(prepared_book, assembly_request.pdf_path,
                                                           assembly_request.book_options)
        fb2_path = output_directory / f"{assembly_request.pdf_path.stem}.fb2"
        fb2_path.write_text(fb2_markup, encoding="utf-8")
        stage_progress.advance_stage(2, ASSEMBLING_STEPS, "validating FB2")
        validation_problems = validate_fb2_file(fb2_path)
        stage_progress.advance_stage(ASSEMBLING_STEPS, ASSEMBLING_STEPS, "FB2 written")
        return fb2_path, {**book_statistics, **proofreading_summary}, validation_problems
