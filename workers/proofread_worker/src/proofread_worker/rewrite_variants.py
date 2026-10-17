"""Text variants of the rewrite tests: the same context followed by each spelling of a word span.

Every variant is ``context + separator + spelling + separator + continuation`` and is measured from the end of
the context, so all variants score their own spelling plus exactly the same following words: the variant with
the highest sum is the most probable reading of the span.
"""

from proofread_worker.deletion_checks import CheckRequest, context_before, continuation_from, separator_between
from proofread_worker.document_text import DocumentText
from worker_protocol.rewrite_checks import RewriteCheck


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
