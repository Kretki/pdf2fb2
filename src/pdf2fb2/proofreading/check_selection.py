"""Which blocks and paragraph breaks the proofreading model should test (structural suspicions).

* every heading (a running head or a stray line labelled as a heading splits the book wrongly);
* every short paragraph (stray words, page furniture, handwriting left over by earlier stages);
* every break between two paragraphs where the first does not end a sentence.
"""

from pdf2fb2.fb2_output.text_markup import SENTENCE_END_PATTERN
from pdf2fb2.proofreading.flow_blocks import FlowBlock

SHORT_BLOCK_WORDS = 6


def block_check_indexes(flow_blocks: list[FlowBlock]) -> list[int]:
    """Blocks to test for deletion.

    Args:
        flow_blocks: Blocks in reading order.

    Returns:
        Indexes of headings and short paragraphs.
    """
    return [block_index for block_index, flow_block in enumerate(flow_blocks)
            if flow_block.block_kind == "heading"
            or (flow_block.block_kind == "p" and len(flow_block.plain_words) <= SHORT_BLOCK_WORDS)]


def break_check_indexes(flow_blocks: list[FlowBlock]) -> list[int]:
    """Second blocks of paragraph breaks that may be one paragraph split in two.

    Args:
        flow_blocks: Blocks in reading order.

    Returns:
        Indexes ``i`` (the break between blocks ``i - 1`` and ``i``).
    """
    return [block_index for block_index in range(1, len(flow_blocks))
            if flow_blocks[block_index].block_kind == "p" and flow_blocks[block_index - 1].block_kind == "p"
            and not SENTENCE_END_PATTERN.search(" ".join(flow_blocks[block_index - 1].plain_words))]
