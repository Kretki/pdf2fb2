"""Unit tests of verifier-gated proofreading: Qwen3.5 examines only the words the token verifier flags."""

from pathlib import Path

from pdf2fb2.fb2_output.flow_model import BookFlow
from pdf2fb2.model_stages.verify_stage import VerifyOutcome
from pdf2fb2.proofreading.edit_decisions import unit_logprob
from pdf2fb2.proofreading.flow_proofreader import LanguageModelProofreader
from pdf2fb2.text_verification.word_flagging import VerifierWordFlagger, unit_word_indexes

THRESHOLDS = {"misread": 0.5, "inserted": 0.5}


class SilentProgress:
    """Progress sink that ignores everything (test double)."""

    def advance_stage(self, done_count: int, total_count: int, step_text: str) -> None:
        """Ignore progress.

        Args:
            done_count: Done units.
            total_count: All units.
            step_text: Step description.

        Returns:
            None.
        """

    def report_message(self, message_text: str) -> None:
        """Ignore messages.

        Args:
            message_text: Message.

        Returns:
            None.
        """


class FlaggingLabeler:
    """Verifier double: ``rnodern`` is misread, everything else correct."""

    def label_units(self, unit_lists: dict[str, list[str]], work_directory: Path, job_name: str,
                    stage_progress: SilentProgress) -> VerifyOutcome:
        """Label the units.

        Args:
            unit_lists: Item key -> units.
            work_directory: Unused.
            job_name: Unused.
            stage_progress: Unused.

        Returns:
            Outcome.
        """
        return VerifyOutcome({item_key: [[0.1, 0.9, 0.0] if unit_text == "rnodern" else [0.95, 0.03, 0.02]
                                         for unit_text in item_units]
                              for item_key, item_units in unit_lists.items()}, THRESHOLDS)


class RecordingScorer:
    """Block scorer double: records the targeted words and returns an empty targeted result."""

    def __init__(self) -> None:
        """Start without calls."""
        self.target_words: list[tuple[int, int]] | None = None

    def score_blocks(self, block_words, block_checks, break_checks, rewrite_checks, work_directory, stage_progress,
                     target_words=None) -> dict:
        """Record the targets.

        Args:
            block_words: Blocks.
            block_checks: Unused.
            break_checks: Unused.
            rewrite_checks: Unused.
            work_directory: Unused.
            stage_progress: Unused.
            target_words: Recorded.

        Returns:
            Worker result without tests.
        """
        self.target_words = target_words
        return {"blocks": [{"words": [{"tokens": [], "min_logprob": 0.0, "mean_logprob": 0.0}
                                      for _ in plain_words]} for _, plain_words in block_words],
                "word_checks": [], "block_checks": [], "break_checks": [], "rewrite_checks": [], "respell_checks": []}


def test_cjk_words_map_every_character_to_their_word() -> None:
    """A Japanese word gives one unit per character, all owned by that word."""
    assert unit_word_indexes(("the", "赤い")) == (["the", "赤", "い"], [0, 1, 1])


def test_only_flagged_words_reach_the_language_model(tmp_path) -> None:
    """The proofreader passes the verifier's flagged words as targets and reports the flagged scope."""
    book_flow = BookFlow(flow_items=[{"kind": "p", "text": "The rnodern age began.", "style": "p", "list": False,
                                      "pages": {1}}])
    recording_scorer = RecordingScorer()
    edit_summary = LanguageModelProofreader(recording_scorer, VerifierWordFlagger(FlaggingLabeler())).proofread_flow(
        book_flow, tmp_path, SilentProgress())
    assert recording_scorer.target_words == [(0, 1)]
    assert edit_summary["proofread_scope"] == "flagged" and edit_summary["flagged_words"] == 1


def test_unit_logprob_uses_the_own_variant_without_tokens() -> None:
    """Targeted results carry ``own``: the unit's log-probability is own - keep."""
    assert unit_logprob({"keep": -10.0, "drop": -12.0, "own": -14.5}, [], 0, range(0, 1)) == -4.5
