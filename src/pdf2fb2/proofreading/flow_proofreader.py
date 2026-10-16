"""Proofreads the book flow with a language model and rearranges it (wiring of the proofreading steps)."""

from pathlib import Path
from typing import Protocol

from pdf2fb2.fb2_output.flow_model import BookFlow
from pdf2fb2.progress.progress_contracts import StageProgress
from pdf2fb2.proofreading.check_selection import block_check_indexes, break_check_indexes
from pdf2fb2.proofreading.edit_decisions import decide_edits
from pdf2fb2.proofreading.flow_blocks import collect_flow_blocks
from pdf2fb2.proofreading.flow_editing import apply_flow_edits
from pdf2fb2.proofreading.proofread_report import write_proofread_report

PROOFREAD_REPORT_FILE = "proofread_report.json"


class BlockScorer(Protocol):
    """Scores text blocks with a language model (see ``model_stages.proofread_stage``)."""

    def score_blocks(self, block_words: list[tuple[str, tuple[str, ...]]], block_checks: list[int],
                     break_checks: list[int], work_directory: Path, stage_progress: StageProgress) -> dict:
        """Token probabilities and deletion-test scores.

        Args:
            block_words: ``(kind, plain words)`` per block.
            block_checks: Blocks to test for deletion.
            break_checks: Blocks whose preceding break is tested.
            work_directory: Output directory.
            stage_progress: Progress sink.

        Returns:
            Worker result document.
        """
        ...


class LanguageModelProofreader:
    """Finds words, headings and paragraph breaks that do not belong to the book and fixes the flow.

    Usage:
        Build with a ``BlockScorer`` and call ``proofread_flow`` before the FB2 is rendered.
    """

    def __init__(self, block_scorer: BlockScorer) -> None:
        """Store the scorer.

        Args:
            block_scorer: Language-model scoring stage.
        """
        self._block_scorer = block_scorer

    def proofread_flow(self, book_flow: BookFlow, work_directory: Path, stage_progress: StageProgress) -> dict:
        """Score the flow text, decide the edits, apply them and write ``proofread_report.json``.

        Args:
            book_flow: Built flow (``flow_items`` replaced by the edited flow).
            work_directory: Output directory.
            stage_progress: Progress sink of the proofreading stage.

        Returns:
            Summary counts of the edits.

        Raises:
            WorkerFailedError: If the language-model worker fails.
        """
        flow_blocks = collect_flow_blocks(book_flow.flow_items)
        if not flow_blocks:
            stage_progress.advance_stage(1, 1, "no text to proofread")
            return {}
        worker_result = self._block_scorer.score_blocks(
            [(flow_block.block_kind, flow_block.plain_words) for flow_block in flow_blocks],
            block_check_indexes(flow_blocks), break_check_indexes(flow_blocks), work_directory, stage_progress)
        flow_edits = decide_edits(flow_blocks, worker_result)
        edit_summary = write_proofread_report(work_directory / PROOFREAD_REPORT_FILE, flow_blocks, worker_result,
                                              flow_edits)
        book_flow.flow_items = apply_flow_edits(book_flow.flow_items, flow_blocks, flow_edits)
        return edit_summary
