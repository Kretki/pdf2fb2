"""Runs a proofreading job: score the whole text, pick suspicious words, run all deletion, rewrite and respelling
tests."""

from collections.abc import Callable

from proofread_worker.deletion_checks import CheckRequest, block_check, break_check, word_check
from proofread_worker.document_lexicon import DocumentLexicon
from proofread_worker.document_text import DocumentText, build_document_text
from proofread_worker.respelling_candidates import candidate_spellings, split_word
from proofread_worker.rewrite_variants import respell_request, rewrite_request
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


def document_lexicon_of(document_text: DocumentText) -> DocumentLexicon:
    """Vocabulary of the document (lower-case word cores).

    Args:
        document_text: Joined document.

    Returns:
        The lexicon.
    """
    return DocumentLexicon([split_word(document_text.full_text[word_span.start_offset:word_span.end_offset])[1].lower()
                            for word_span in document_text.word_spans])


def respell_requests(document_text: DocumentText, proofread_settings: ProofreadSettings,
                     word_span_indexes: list[int]) -> list[CheckRequest]:
    """Respelling tests of the least probable words.

    Args:
        document_text: Joined document.
        proofread_settings: Candidate count and the cap on the number of tests.
        word_span_indexes: Suspicious words, least probable first.

    Returns:
        Requests for the words that have candidate spellings (at most ``max_respell_checks``).
    """
    if not proofread_settings.max_respell_checks:
        return []
    document_lexicon = document_lexicon_of(document_text)
    test_limits = (proofread_settings.context_characters, proofread_settings.continuation_words)
    check_requests: list[CheckRequest] = []
    for span_index in word_span_indexes:
        word_span = document_text.word_spans[span_index]
        spelling_candidates = candidate_spellings(document_text.full_text[word_span.start_offset:word_span.end_offset],
                                                  document_lexicon, proofread_settings.respell_candidates)
        check_request = respell_request(document_text, span_index, spelling_candidates, *test_limits)
        if check_request is not None:
            check_requests.append(check_request)
        if len(check_requests) >= proofread_settings.max_respell_checks:
            break
    return check_requests


def target_spans(document_text: DocumentText, proofread_job: ProofreadJob) -> list[int]:
    """Word spans of the words flagged by the token verifier.

    Args:
        document_text: Joined document.
        proofread_job: Targeted job.

    Returns:
        Span indexes in document order, at most ``max_word_checks``.
    """
    flagged_spans = sorted({document_text.block_word_offsets[block_index] + word_index
                            for block_index, word_index in proofread_job.target_words or ()})
    return flagged_spans[:proofread_job.proofread_settings.max_word_checks]


def build_check_requests(document_text: DocumentText, proofread_job: ProofreadJob,
                         word_span_indexes: list[int]) -> list[CheckRequest]:
    """All deletion, rewrite and respelling tests of the job.

    Args:
        document_text: Joined document.
        proofread_job: Job with the requested block and break tests.
        word_span_indexes: Suspicious words.

    Returns:
        Requests (impossible tests, e.g. at the document end, are left out).
    """
    proofread_settings = proofread_job.proofread_settings
    test_limits = (proofread_settings.context_characters, proofread_settings.continuation_words)
    include_own = proofread_job.target_words is not None
    check_requests = [word_check(document_text, span_index, *test_limits, include_own)
                      for span_index in word_span_indexes]
    check_requests += [block_check(document_text, block_index, *test_limits, include_own)
                       for block_index in proofread_job.block_checks]
    check_requests += [break_check(document_text, block_index, *test_limits)
                       for block_index in proofread_job.break_checks]
    check_requests += [rewrite_request(document_text, rewrite_check, *test_limits)
                       for rewrite_check in proofread_job.rewrite_checks]
    check_requests += respell_requests(document_text, proofread_settings, word_span_indexes)
    return [check_request for check_request in check_requests if check_request is not None]


def run_check_requests(token_scorer: TokenScorer, check_requests: list[CheckRequest],
                       progress_callback: ProgressCallback) -> dict[str, list[dict]]:
    """Score every variant of every test.

    Args:
        token_scorer: Model wrapper.
        check_requests: Tests.
        progress_callback: Receives ``("checking", done, total)``.

    Returns:
        ``{"word_checks": [...], "block_checks": [...], "break_checks": [...], "rewrite_checks": [...],
        "respell_checks": [...]}``.
    """
    flat_variants = [scored_variant for check_request in check_requests
                     for scored_variant in check_request.scored_variants]
    variant_scores = token_scorer.continuation_logprobs(
        flat_variants, lambda done_count, total_count: progress_callback("checking", done_count, total_count))
    grouped_results: dict[str, list[dict]] = {"word_checks": [], "block_checks": [], "break_checks": [],
                                              "rewrite_checks": [], "respell_checks": []}
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
    """Score the book text (whole-text jobs only) and run the deletion, rewrite and respelling tests.

    Args:
        proofread_job: Validated job.
        token_scorer: Model wrapper.
        progress_callback: Receives ``(phase, done, total)`` with phase ``scoring`` or ``checking``.

    Returns:
        Result document (see ``worker_protocol.proofread_job_schema``).
    """
    document_text = build_document_text(proofread_job.text_blocks)
    if proofread_job.target_words is None:
        scored_tokens = token_scorer.score_document(document_text.full_text, lambda done_count, total_count:
                                                    progress_callback("scoring", done_count, total_count))
        word_tokens = tokens_by_word(document_text, scored_tokens)
        word_span_indexes = candidate_spans(word_tokens, proofread_job.proofread_settings)
    else:
        word_tokens = [[] for _ in document_text.word_spans]
        word_span_indexes = target_spans(document_text, proofread_job)
    check_requests = build_check_requests(document_text, proofread_job, word_span_indexes)
    return {"blocks": block_word_entries(document_text, word_tokens, len(proofread_job.text_blocks)),
            **run_check_requests(token_scorer, check_requests, progress_callback)}
