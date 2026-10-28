"""Proofreads the book flow with a language model and rearranges it (wiring of the proofreading steps).

With a word flagger (the token verifier) the language model no longer reads the whole book: it examines only the
flagged words (deletion and respelling tests) plus the structural tests (headings, short paragraphs, paragraph
breaks, hyphen breaks), each a short context-and-continuation test. Without one, every token is scored and every
word below e^-9 is tested (full mode).
"""

from pathlib import Path
from typing import Protocol

from pdf2fb2.fb2_output.flow_model import BookFlow
from pdf2fb2.progress.progress_contracts import StageProgress
from pdf2fb2.proofreading.check_selection import block_check_indexes, break_check_indexes
from pdf2fb2.proofreading.edit_decisions import decide_edits
from pdf2fb2.proofreading.flow_blocks import collect_flow_blocks
from pdf2fb2.proofreading.flow_editing import apply_flow_edits
from pdf2fb2.proofreading.hyphen_repair import find_hyphen_breaks, rewrite_checks_for
from pdf2fb2.proofreading.proofread_report import write_proofread_report
from worker_protocol.rewrite_checks import RewriteCheck

PROOFREAD_REPORT_FILE = "proofread_report.json"


class BlockScorer(Protocol):
    """Scores text blocks with a language model (see ``model_stages.proofread_stage``)."""

    def score_blocks(self, block_words: list[tuple[str, tuple[str, ...]]], block_checks: list[int],
                     break_checks: list[int], rewrite_checks: list[RewriteCheck], work_directory: Path,
                     stage_progress: StageProgress, target_words: list[tuple[int, int]] | None = None) -> dict:
        """Token probabilities, deletion-test, rewrite-test and respelling-test scores.

        Args:
            block_words: ``(kind, plain words)`` per block.
            block_checks: Blocks to test for deletion.
            break_checks: Blocks whose preceding break is tested.
            rewrite_checks: Word spans whose spellings are compared.
            work_directory: Output directory.
            stage_progress: Progress sink.
            target_words: Words to examine (None: score the whole text).

        Returns:
            Worker result document.
        """
        ...


class WordFlagger(Protocol):
    """Selects the words the language model examines (see ``text_verification.word_flagging``)."""

    def flag_words(self, block_words: list[tuple[str, tuple[str, ...]]], work_directory: Path,
                   stage_progress: StageProgress) -> list[tuple[int, int]]:
        """Flagged words.

        Args:
            block_words: ``(kind, plain words)`` per block.
            work_directory: Output directory.
            stage_progress: Progress sink.

        Returns:
            ``(block, word)`` pairs.
        """
        ...


class LanguageModelProofreader:
    """Finds words, headings and paragraph breaks that do not belong to the book and fixes the flow.

    Usage:
        Build with a ``BlockScorer`` (and optionally a ``WordFlagger``) and call ``proofread_flow`` before the
        FB2 is rendered.
    """

    def __init__(self, block_scorer: BlockScorer, word_flagger: WordFlagger | None = None) -> None:
        """Store the stages.

        Args:
            block_scorer: Language-model scoring stage.
            word_flagger: Token verifier choosing the examined words; None runs the full mode.
        """
        self._block_scorer = block_scorer
        self._word_flagger = word_flagger

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
        block_words = [(flow_block.block_kind, flow_block.plain_words) for flow_block in flow_blocks]
        target_words = None if self._word_flagger is None else self._word_flagger.flag_words(
            block_words, work_directory, stage_progress)
        worker_result = self._block_scorer.score_blocks(
            block_words, block_check_indexes(flow_blocks), break_check_indexes(flow_blocks),
            rewrite_checks_for(flow_blocks, find_hyphen_breaks(flow_blocks)), work_directory, stage_progress,
            target_words)
        flow_edits = decide_edits(flow_blocks, worker_result)
        edit_summary = write_proofread_report(work_directory / PROOFREAD_REPORT_FILE, flow_blocks, worker_result,
                                              flow_edits)
        book_flow.flow_items = apply_flow_edits(book_flow.flow_items, flow_blocks, flow_edits)
        total_words = sum(len(flow_block.plain_words) for flow_block in flow_blocks)
        scope_summary = {"proofread_scope": "full"} if target_words is None else {
            "proofread_scope": "flagged", "flagged_words": len(target_words), "flow_words": total_words}
        return {**edit_summary, **scope_summary}
