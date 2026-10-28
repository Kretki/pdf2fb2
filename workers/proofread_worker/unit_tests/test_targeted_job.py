"""Unit tests of targeted proofreading jobs: only verifier-flagged words, no whole-document scoring."""

from proofread_worker.deletion_checks import block_check, word_check
from proofread_worker.document_text import build_document_text
from proofread_worker.job_execution import run_proofread_job
from worker_protocol.proofread_job_schema import ProofreadJob, ProofreadSettings, TextBlock

SAMPLE_BLOCKS = (TextBlock("p", ("He", "rode", "to", "the", "old", "rnission", "at", "dawn.")),
                 TextBlock("heading", ("Running", "Head")), TextBlock("p", ("Then", "he", "slept.")))


class CountingScorer:
    """Language-model double: every measured character costs -0.1; whole-document scoring is forbidden."""

    def __init__(self) -> None:
        """Start with no calls."""
        self.scored_variants: list[tuple[str, int]] = []

    def score_document(self, document_text: str, batch_callback) -> list:
        """Fail: targeted jobs must not score the whole text.

        Args:
            document_text: Unused.
            batch_callback: Unused.

        Raises:
            AssertionError: Always.
        """
        raise AssertionError("whole-document scoring in a targeted job")

    def continuation_logprobs(self, scored_texts: list[tuple[str, int]], batch_callback) -> list[float]:
        """Score the measured part by its length.

        Args:
            scored_texts: ``(text, measure_from)`` pairs.
            batch_callback: Unused.

        Returns:
            One score per text.
        """
        self.scored_variants.extend(scored_texts)
        return [-0.1 * (len(scored_text) - measure_from) for scored_text, measure_from in scored_texts]


def test_own_variant_measures_the_unit_and_its_continuation() -> None:
    """``own`` keeps the kept text and starts measuring at the unit, so own - keep is the unit itself."""
    document_text = build_document_text(SAMPLE_BLOCKS)
    word_request = word_check(document_text, 5, 200, 2, include_own=True)
    assert word_request.variant_names == ("keep", "drop", "own")
    own_text, own_start = word_request.scored_variants[2]
    assert own_text[own_start:] == "rnission at dawn."  # the token " rnission" ends after own_start: counted
    block_request = block_check(document_text, 1, 200, 2, include_own=True)
    assert block_request.variant_names[-1] == "own"
    assert block_request.scored_variants[3][0][block_request.scored_variants[3][1]:].startswith("Running Head")


def test_targeted_job_tests_only_flagged_words(tmp_path) -> None:
    """Only the flagged word is word-tested and respelled; blocks come back without tokens."""
    proofread_job = ProofreadJob(tmp_path, tmp_path, "cpu", SAMPLE_BLOCKS, (1,), (), ProofreadSettings(
        -9.0, 200, 2, 100, max_respell_checks=10, respell_candidates=2), "result.json", target_words=((0, 5),))
    decoded_job = ProofreadJob.from_payload(proofread_job.to_payload())
    assert decoded_job.target_words == ((0, 5),)
    counting_scorer = CountingScorer()
    worker_result = run_proofread_job(decoded_job, counting_scorer, lambda phase_name, done_count, total_count: None)
    assert [(word_entry["block"], word_entry["word"]) for word_entry in worker_result["word_checks"]] == [(0, 5)]
    assert "own" in worker_result["word_checks"][0] and "own" in worker_result["block_checks"][0]
    assert all(not word_result["tokens"] for block_result in worker_result["blocks"]
               for word_result in block_result["words"])
    assert all(respell_entry["word"] == 5 for respell_entry in worker_result["respell_checks"])
