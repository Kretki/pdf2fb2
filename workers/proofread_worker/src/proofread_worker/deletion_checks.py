"""Text variants of the deletion tests: the same continuation scored with and without a unit.

Every variant is ``context + [unit] + separator + continuation`` and is measured from the end of the
unit (or of the context when the unit is dropped), so all variants of one test score exactly the
same following words, including the separator that leads into them.
"""

from dataclasses import dataclass

from proofread_worker.document_text import BLOCK_SEPARATOR, WORD_SEPARATOR, DocumentText

ScoredVariant = tuple[str, int]


@dataclass(frozen=True)
class CheckRequest:
    """One test: what it is about and the variants to score.

    Attributes:
        check_kind: ``word``, ``block`` or ``break``.
        check_target: ``{"block": i}`` or ``{"block": i, "word": j}``.
        variant_names: Result keys, e.g. ``("keep", "drop")``.
        scored_variants: ``(text, measure_from)`` per variant name.
    """

    check_kind: str
    check_target: dict
    variant_names: tuple[str, ...]
    scored_variants: tuple[ScoredVariant, ...]


def context_before(document_text: DocumentText, end_offset: int, context_characters: int) -> str:
    """Preceding text, cut at a whitespace boundary.

    Args:
        document_text: Document.
        end_offset: Offset where the context ends.
        context_characters: Maximum length.

    Returns:
        Context string (may be empty).
    """
    start_offset = max(0, end_offset - context_characters)
    if start_offset > 0:
        boundary = document_text.full_text.find(WORD_SEPARATOR, start_offset, end_offset)
        start_offset = boundary + 1 if boundary >= 0 else start_offset
    return document_text.full_text[start_offset:end_offset]


def continuation_from(document_text: DocumentText, span_index: int, word_count: int) -> str:
    """``word_count`` words of the document starting at a word span.

    Args:
        document_text: Document.
        span_index: First word span.
        word_count: Number of words.

    Returns:
        Text from the first to the last of those words (separators included).
    """
    last_index = min(len(document_text.word_spans) - 1, span_index + word_count - 1)
    return document_text.full_text[document_text.word_spans[span_index].start_offset:
                                   document_text.word_spans[last_index].end_offset]


def separator_between(document_text: DocumentText, left_index: int, right_index: int) -> str:
    """Text between two word spans (a space or a blank line).

    Args:
        document_text: Document.
        left_index: Earlier span, or -1 for the document start.
        right_index: Later span.

    Returns:
        Separator text.
    """
    left_end = document_text.word_spans[left_index].end_offset if left_index >= 0 else 0
    return document_text.full_text[left_end:document_text.word_spans[right_index].start_offset]


def word_check(document_text: DocumentText, span_index: int, context_characters: int,
               continuation_words: int) -> CheckRequest | None:
    """Test whether the words after a word read better without it.

    Args:
        document_text: Document.
        span_index: Tested word span.
        context_characters: Context length.
        continuation_words: Measured words after the unit.

    Returns:
        The request, or None for the last word of the document.
    """
    if span_index + 1 >= len(document_text.word_spans):
        return None
    word_span = document_text.word_spans[span_index]
    previous_end = document_text.word_spans[span_index - 1].end_offset if span_index else 0
    context_text = context_before(document_text, previous_end, context_characters)
    separator_before = separator_between(document_text, span_index - 1, span_index)
    separator_after = separator_between(document_text, span_index, span_index + 1)
    continuation_text = continuation_from(document_text, span_index + 1, continuation_words)
    word_text = document_text.full_text[word_span.start_offset:word_span.end_offset]
    kept_head = context_text + separator_before + word_text
    joined_separator = BLOCK_SEPARATOR if BLOCK_SEPARATOR in (separator_before, separator_after) else WORD_SEPARATOR
    joined_separator = joined_separator if context_text else ""
    return CheckRequest("word", {"block": word_span.block_index, "word": word_span.word_index}, ("keep", "drop"),
                        ((kept_head + separator_after + continuation_text, len(kept_head)),
                         (context_text + joined_separator + continuation_text, len(context_text))))


def block_check(document_text: DocumentText, block_index: int, context_characters: int,
                continuation_words: int) -> CheckRequest | None:
    """Test whether the text after a block reads better without it, separately or merged.

    Args:
        document_text: Document.
        block_index: Tested block.
        context_characters: Context length.
        continuation_words: Measured words after the block.

    Returns:
        The request, or None for empty blocks and the last block.
    """
    first_span = document_text.block_word_offsets[block_index]
    block_start, block_end = document_text.block_bounds[block_index]
    if block_start == block_end or block_index + 1 >= len(document_text.block_bounds):
        return None
    next_span = document_text.block_word_offsets[block_index + 1]
    if next_span >= len(document_text.word_spans):
        return None
    previous_end = document_text.word_spans[first_span - 1].end_offset if first_span else 0
    context_text = context_before(document_text, previous_end, context_characters)
    lead_separator = BLOCK_SEPARATOR if context_text else ""
    kept_head = context_text + lead_separator + document_text.full_text[block_start:block_end]
    continuation_text = separator_between(document_text, next_span - 1, next_span) + \
        continuation_from(document_text, next_span, continuation_words)
    following_text = continuation_text.lstrip()
    merged_separator = WORD_SEPARATOR if context_text else ""
    return CheckRequest("block", {"block": block_index}, ("keep", "drop_separate", "drop_merged"),
                        ((kept_head + continuation_text, len(kept_head)),
                         (context_text + lead_separator + following_text, len(context_text)),
                         (context_text + merged_separator + following_text, len(context_text))))


def break_check(document_text: DocumentText, block_index: int, context_characters: int,
                continuation_words: int) -> CheckRequest | None:
    """Test whether a block continues the previous one (a paragraph wrongly split in two).

    Args:
        document_text: Document.
        block_index: Second block of the break.
        context_characters: Context length.
        continuation_words: Measured words of the second block.

    Returns:
        The request, or None for the first block or empty blocks.
    """
    first_span = document_text.block_word_offsets[block_index]
    block_start, block_end = document_text.block_bounds[block_index]
    if block_index == 0 or block_start == block_end or first_span == 0:
        return None
    context_text = context_before(document_text, document_text.word_spans[first_span - 1].end_offset,
                                  context_characters)
    continuation_text = continuation_from(document_text, first_span, continuation_words)
    return CheckRequest("break", {"block": block_index}, ("keep", "merged"),
                        ((context_text + BLOCK_SEPARATOR + continuation_text, len(context_text)),
                         (context_text + WORD_SEPARATOR + continuation_text, len(context_text))))
