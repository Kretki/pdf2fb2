"""Runs a proofreading job: score the whole text, pick suspicious words, run all deletion tests."""

from collections.abc import Callable

from proofread_worker.deletion_checks import CheckRequest, block_check, break_check, word_check
from proofread_worker.document_text import DocumentText, build_document_text
from proofread_worker.token_scorer import TokenScorer
from proofread_worker.word_statistics import block_word_entries, tokens_by_word
from worker_protocol.proofread_job_schema import ProofreadJob, ProofreadSettings

ProgressCallback = Callable[[str, int, int], None]
RESULT_DIGITS = 3


def candidate_spans(word_tokens: list[list], proofread_settings: ProofreadSettings) -> list[int]:
    """Word spans whose least probable token is below the threshold, least probable first.

    Args:
        word_tokens: Scored tokens per word span.
        proofread_settings: Threshold and the cap on the number of tests.

    Returns:
        Span indexes to test.
    """
    minimum_logprobs = [(min(scored_token.token_logprob for scored_token in scored_tokens), span_index)
                        for span_index, scored_tokens in enumerate(word_tokens) if scored_tokens]
    suspicious_words = sorted(minimum_entry for minimum_entry in minimum_logprobs
                              if minimum_entry[0] < proofread_settings.candidate_logprob)
    return [span_index for _, span_index in suspicious_words[:proofread_settings.max_word_checks]]


def build_check_requests(document_text: DocumentText, proofread_job: ProofreadJob,
                         word_span_indexes: list[int]) -> list[CheckRequest]:
    """All deletion tests of the job.

    Args:
        document_text: Joined document.
        proofread_job: Job with the requested block and break tests.
        word_span_indexes: Suspicious words.

    Returns:
        Requests (impossible tests, e.g. at the document end, are left out).
    """
    proofread_settings = proofread_job.proofread_settings
    test_limits = (proofread_settings.context_characters, proofread_settings.continuation_words)
    check_requests = [word_check(document_text, span_index, *test_limits) for span_index in word_span_indexes]
    check_requests += [block_check(document_text, block_index, *test_limits)
                       for block_index in proofread_job.block_checks]
    check_requests += [break_check(document_text, block_index, *test_limits)
                       for block_index in proofread_job.break_checks]
    return [check_request for check_request in check_requests if check_request is not None]


def run_check_requests(token_scorer: TokenScorer, check_requests: list[CheckRequest],
                       progress_callback: ProgressCallback) -> dict[str, list[dict]]:
    """Score every variant of every test.

    Args:
        token_scorer: Model wrapper.
        check_requests: Tests.
        progress_callback: Receives ``("checking", done, total)``.

    Returns:
        ``{"word_checks": [...], "block_checks": [...], "break_checks": [...]}``.
    """
    flat_variants = [scored_variant for check_request in check_requests
                     for scored_variant in check_request.scored_variants]
    variant_scores = token_scorer.continuation_logprobs(
        flat_variants, lambda done_count, total_count: progress_callback("checking", done_count, total_count))
    grouped_results: dict[str, list[dict]] = {"word_checks": [], "block_checks": [], "break_checks": []}
    read_offset = 0
    for check_request in check_requests:
        request_scores = variant_scores[read_offset:read_offset + len(check_request.scored_variants)]
        read_offset += len(check_request.scored_variants)
        grouped_results[f"{check_request.check_kind}_checks"].append(
            {**check_request.check_target, **{variant_name: round(variant_score, RESULT_DIGITS) for
                                              variant_name, variant_score in zip(check_request.variant_names,
                                                                                 request_scores, strict=True)}})
    return grouped_results


def run_proofread_job(proofread_job: ProofreadJob, token_scorer: TokenScorer,
                      progress_callback: ProgressCallback) -> dict:
    """Score the book text and run the deletion tests.

    Args:
        proofread_job: Validated job.
        token_scorer: Model wrapper.
        progress_callback: Receives ``(phase, done, total)`` with phase ``scoring`` or ``checking``.

    Returns:
        Result document (see ``worker_protocol.proofread_job_schema``).
    """
    document_text = build_document_text(proofread_job.text_blocks)
    scored_tokens = token_scorer.score_document(
        document_text.full_text, lambda done_count, total_count: progress_callback("scoring", done_count, total_count))
    word_tokens = tokens_by_word(document_text, scored_tokens)
    check_requests = build_check_requests(document_text, proofread_job,
                                          candidate_spans(word_tokens, proofread_job.proofread_settings))
    return {"blocks": block_word_entries(document_text, word_tokens, len(proofread_job.text_blocks)),
            **run_check_requests(token_scorer, check_requests, progress_callback)}
