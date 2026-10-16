"""The book text as one string: blocks separated by a blank line, words by a space, with word offsets."""

from bisect import bisect_right
from dataclasses import dataclass

from worker_protocol.proofread_job_schema import TextBlock

BLOCK_SEPARATOR = "\n\n"
WORD_SEPARATOR = " "


@dataclass(frozen=True)
class WordSpan:
    """Position of one word in the document string.

    Attributes:
        block_index: Index of the block.
        word_index: Index of the word inside the block.
        start_offset: First character.
        end_offset: Character after the word.
    """

    block_index: int
    word_index: int
    start_offset: int
    end_offset: int


@dataclass(frozen=True)
class DocumentText:
    """Joined text plus the spans of all words and blocks.

    Attributes:
        full_text: Document string.
        word_spans: Spans in document order.
        block_bounds: ``(start, end)`` character range of each block (empty blocks have start == end).
        block_word_offsets: Index into ``word_spans`` of the first word of each block.
        word_starts: Start offset of every word span (for binary search).
    """

    full_text: str
    word_spans: tuple[WordSpan, ...]
    block_bounds: tuple[tuple[int, int], ...]
    block_word_offsets: tuple[int, ...]
    word_starts: tuple[int, ...]

    def span_containing(self, character_offset: int) -> int | None:
        """Index of the word span that contains a character.

        Args:
            character_offset: Offset in ``full_text``.

        Returns:
            Index into ``word_spans`` or None for separator characters.
        """
        span_index = bisect_right(self.word_starts, character_offset) - 1
        if span_index < 0 or character_offset >= self.word_spans[span_index].end_offset:
            return None
        return span_index


def build_document_text(text_blocks: tuple[TextBlock, ...]) -> DocumentText:
    """Join the blocks into one string and record where every word lies.

    Usage:
        Call once per job; every test is built from slices of this string.

    Args:
        text_blocks: Blocks in reading order.

    Returns:
        The document text.
    """
    text_pieces: list[str] = []
    word_spans: list[WordSpan] = []
    block_bounds: list[tuple[int, int]] = []
    block_word_offsets: list[int] = []
    current_offset = 0
    for block_index, text_block in enumerate(text_blocks):
        if text_pieces:
            text_pieces.append(BLOCK_SEPARATOR)
            current_offset += len(BLOCK_SEPARATOR)
        block_start = current_offset
        block_word_offsets.append(len(word_spans))
        for word_index, word_text in enumerate(text_block.block_words):
            if word_index:
                text_pieces.append(WORD_SEPARATOR)
                current_offset += len(WORD_SEPARATOR)
            word_spans.append(WordSpan(block_index, word_index, current_offset, current_offset + len(word_text)))
            text_pieces.append(word_text)
            current_offset += len(word_text)
        block_bounds.append((block_start, current_offset))
    return DocumentText("".join(text_pieces), tuple(word_spans), tuple(block_bounds), tuple(block_word_offsets),
                        tuple(word_span.start_offset for word_span in word_spans))
