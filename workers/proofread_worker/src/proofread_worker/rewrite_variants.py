"""Text variants of the rewrite tests: the same context followed by each spelling of a word span.

Every variant is ``context + separator + spelling + separator + continuation`` and is measured from the end of
the context, so all variants score their own spelling plus exactly the same following words: the variant with
the highest sum is the most probable reading of the span. Respelling tests are rewrite tests of one
low-probability word whose spellings come from ``respelling_candidates``.
"""

from dataclasses import replace

from proofread_worker.deletion_checks import CheckRequest, context_before, continuation_from, separator_between
from proofread_worker.document_text import DocumentText
from proofread_worker.respelling_candidates import RespellCandidate
from worker_protocol.rewrite_checks import RewriteCheck

KEPT_SPELLING = "kept"
RESPELL_NAMES = ("alt_a", "alt_b", "alt_c", "alt_d", "alt_e", "alt_f", "alt_g")


def rewrite_request(document_text: DocumentText, rewrite_check: RewriteCheck, context_characters: int,
                    continuation_words: int) -> CheckRequest:
    """Build the variants of one rewrite test.

    Args:
        document_text: Document.
        rewrite_check: Span and spellings (validated against the job's blocks).
        context_characters: Context length.
        continuation_words: Measured words after the span.

    Returns:
        The request; its variant names are the spelling names.
    """
    first_span = document_text.block_word_offsets[rewrite_check.block_index] + rewrite_check.word_index
    last_span = first_span + rewrite_check.word_count - 1
    previous_end = document_text.word_spans[first_span - 1].end_offset if first_span else 0
    context_text = context_before(document_text, previous_end, context_characters)
    separator_before = separator_between(document_text, first_span - 1, first_span) if context_text else ""
    following_text = ""
    if last_span + 1 < len(document_text.word_spans):
        following_text = separator_between(document_text, last_span, last_span + 1) + continuation_from(
            document_text, last_span + 1, continuation_words)
    return CheckRequest(
        "rewrite", {"block": rewrite_check.block_index, "word": rewrite_check.word_index,
                    "count": rewrite_check.word_count},
        tuple(variant_name for variant_name, _ in rewrite_check.variant_texts),
        tuple((context_text + separator_before + variant_text + following_text, len(context_text))
              for _, variant_text in rewrite_check.variant_texts))


def respell_request(document_text: DocumentText, span_index: int, respell_candidates: list[RespellCandidate],
                    context_characters: int, continuation_words: int) -> CheckRequest | None:
    """Build the variants of one respelling test: the word as read and its candidate spellings.

    Args:
        document_text: Document.
        span_index: Word span to respell.
        respell_candidates: Alternative spellings, best first.
        context_characters: Context length.
        continuation_words: Measured words after the word.

    Returns:
        A ``respell`` request whose target carries the spellings and their edit costs, or None without candidates.
    """
    if not respell_candidates:
        return None
    word_span = document_text.word_spans[span_index]
    word_text = document_text.full_text[word_span.start_offset:word_span.end_offset]
    variant_names = (KEPT_SPELLING, *RESPELL_NAMES[:len(respell_candidates)])
    variant_texts = tuple(zip(variant_names, (word_text, *(respell_candidate.spelling for respell_candidate
                                                            in respell_candidates)), strict=True))
    rewrite_check = RewriteCheck(word_span.block_index, word_span.word_index, 1, variant_texts)
    rewrite_variants = rewrite_request(document_text, rewrite_check, context_characters, continuation_words)
    return replace(rewrite_variants, check_kind="respell", check_target={
        "block": word_span.block_index, "word": word_span.word_index, "spellings": dict(variant_texts[1:]),
        "costs": dict(zip(variant_names[1:], (respell_candidate.edit_cost for respell_candidate
                                              in respell_candidates), strict=True))})
