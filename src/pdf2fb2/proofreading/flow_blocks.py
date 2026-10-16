"""Text blocks of the book flow as the proofreading model sees them, with the way back to the flow.

Paragraphs, headings and captions become blocks; their whitespace tokens become words with the
inline markup removed (style tags, note links shown as their marker). Pictures, tables and page
markers are not text and are skipped.
"""

import re
from dataclasses import dataclass

from pdf2fb2.fb2_output.text_markup import NOTE_TOKEN_PATTERN, visible_text

TEXT_ITEM_KINDS = frozenset({"p", "heading", "caption"})
TOKEN_PATTERN = re.compile(r"\S+")


@dataclass(frozen=True)
class FlowBlock:
    """One text item of the flow split into tokens.

    Attributes:
        flow_index: Index of the item in ``flow_items``.
        block_kind: ``p``, ``heading`` or ``caption``.
        raw_tokens: Whitespace tokens of the item text, markup included.
        plain_words: Model words, one per raw token that has visible text.
        word_tokens: Index into ``raw_tokens`` of every plain word.
        page_top: Whether the item directly follows a page marker (where running heads sit).
    """

    flow_index: int
    block_kind: str
    raw_tokens: tuple[str, ...]
    plain_words: tuple[str, ...]
    word_tokens: tuple[int, ...]
    page_top: bool


def plain_word(raw_token: str) -> str:
    """Visible text of a token: style tags removed, note links replaced by their marker.

    Args:
        raw_token: Token of a flow item text.

    Returns:
        Plain text (may be empty for markup-only tokens).
    """
    return visible_text(NOTE_TOKEN_PATTERN.sub(lambda note_match: note_match.group(2), raw_token)).strip()


def flow_block(flow_index: int, flow_item: dict, page_top: bool) -> FlowBlock:
    """Split one flow item into tokens and model words.

    Args:
        flow_index: Index of the item.
        flow_item: Paragraph, heading or caption item.
        page_top: Whether the previous flow item is a page marker.

    Returns:
        The block.
    """
    raw_tokens = tuple(TOKEN_PATTERN.findall(flow_item["text"]))
    word_pairs = [(token_index, plain_word(raw_token)) for token_index, raw_token in enumerate(raw_tokens)]
    word_pairs = [(token_index, word_text) for token_index, word_text in word_pairs if word_text]
    return FlowBlock(flow_index=flow_index, block_kind=flow_item["kind"], raw_tokens=raw_tokens,
                     plain_words=tuple(word_text for _, word_text in word_pairs),
                     word_tokens=tuple(token_index for token_index, _ in word_pairs), page_top=page_top)


def collect_flow_blocks(flow_items: list[dict]) -> list[FlowBlock]:
    """All text blocks of the flow that contain at least one word.

    Usage:
        Call once before proofreading; block indexes in the worker job refer to this list.

    Args:
        flow_items: Book flow items.

    Returns:
        Blocks in reading order.
    """
    text_blocks = [flow_block(flow_index, flow_item, flow_index > 0 and flow_items[flow_index - 1]["kind"] == "page")
                   for flow_index, flow_item in enumerate(flow_items)
                   if flow_item["kind"] in TEXT_ITEM_KINDS and flow_item.get("text")]
    return [text_block for text_block in text_blocks if text_block.plain_words]
