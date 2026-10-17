"""Apply proofreading edits to the book flow: delete words and blocks, respell hyphen breaks, re-join paragraphs.

Deleting a wrong heading removes the section split it caused (sections are built from headings
when the FB2 is rendered); a paragraph marked as merged is appended to the previous surviving
paragraph, across page markers, undoing end-of-line hyphenation.
"""

from pdf2fb2.fb2_output.text_markup import join_text_fragments
from pdf2fb2.proofreading.edit_decisions import FlowEdits
from pdf2fb2.proofreading.flow_blocks import FlowBlock
from pdf2fb2.proofreading.hyphen_repair import HyphenFix, is_hyphen, repaired_token


def respelled_tokens(flow_block: FlowBlock, hyphen_fixes: list[HyphenFix],
                     deleted_words: frozenset[tuple[int, int]]) -> dict[int, str | None]:
    """Raw-token replacements of the hyphen fixes of one block.

    Args:
        flow_block: Block of the item.
        hyphen_fixes: Fixes of this block.
        deleted_words: ``(block_index, word_index)`` pairs (a fix touching a deleted word is skipped).

    Returns:
        Mapping raw token index -> new text (None: token removed); fixes whose words are not adjacent raw
        tokens are skipped.
    """
    replacements: dict[int, str | None] = {}
    for hyphen_fix in hyphen_fixes:
        word_range = range(hyphen_fix.word_index, hyphen_fix.word_index + hyphen_fix.word_count)
        token_indexes = [flow_block.word_tokens[word_index] for word_index in word_range]
        if any((hyphen_fix.block_index, word_index) in deleted_words for word_index in word_range) or \
                token_indexes != list(range(token_indexes[0], token_indexes[0] + len(token_indexes))):
            continue
        left_word = flow_block.plain_words[hyphen_fix.word_index]
        replacements[token_indexes[0]] = repaired_token([flow_block.raw_tokens[token_index]
                                                         for token_index in token_indexes],
                                                        hyphen_fix.variant_name, is_hyphen(left_word[-1:]))
        replacements.update(dict.fromkeys(token_indexes[1:]))
    return replacements


def edited_block_text(flow_block: FlowBlock, block_index: int, flow_edits: FlowEdits) -> str | None:
    """Item text with the deleted words removed and the hyphen breaks respelled.

    Args:
        flow_block: Block of the item.
        block_index: Index of the block.
        flow_edits: Decided edits.

    Returns:
        The resulting tokens joined by spaces (``None`` when the block is not edited).
    """
    replacements = respelled_tokens(flow_block, [hyphen_fix for hyphen_fix in flow_edits.hyphen_fixes
                                                 if hyphen_fix.block_index == block_index], flow_edits.deleted_words)
    replacements.update(dict.fromkeys(flow_block.word_tokens[word_index]
                                      for word_index in range(len(flow_block.word_tokens))
                                      if (block_index, word_index) in flow_edits.deleted_words))
    if not replacements:
        return None
    return " ".join(replacements.get(token_index, raw_token) for token_index, raw_token
                    in enumerate(flow_block.raw_tokens) if replacements.get(token_index, raw_token) is not None)


def merge_into(previous_item: dict, flow_item: dict) -> None:
    """Append a paragraph to the previous one.

    Args:
        previous_item: Surviving paragraph (mutated).
        flow_item: Paragraph that continues it.

    Returns:
        None.
    """
    previous_item["text"] = join_text_fragments(previous_item["text"], flow_item["text"])
    previous_item["pages"] = set(previous_item.get("pages", set())) | set(flow_item.get("pages", set()))


def apply_flow_edits(flow_items: list[dict], flow_blocks: list[FlowBlock], flow_edits: FlowEdits) -> list[dict]:
    """New flow with the edits applied.

    Usage:
        Call after ``decide_edits``; the result replaces ``BookFlow.flow_items``.

    Args:
        flow_items: Original flow (items are copied, not mutated).
        flow_blocks: Blocks the edits refer to.
        flow_edits: Decided edits.

    Returns:
        Edited flow items.
    """
    block_of_item = {flow_block.flow_index: block_index for block_index, flow_block in enumerate(flow_blocks)}
    edited_items: list[dict] = []
    last_paragraph: dict | None = None
    for flow_index, flow_item in enumerate(flow_items):
        block_index = block_of_item.get(flow_index)
        if block_index is None:
            edited_items.append(dict(flow_item))
            last_paragraph = last_paragraph if flow_item["kind"] == "page" else None
            continue
        if block_index in flow_edits.deleted_blocks:
            continue
        edited_text = edited_block_text(flow_blocks[block_index], block_index, flow_edits)
        edited_item = {**flow_item, "text": flow_item["text"] if edited_text is None else edited_text}
        if block_index in flow_edits.merged_blocks and last_paragraph is not None and edited_item["kind"] == "p" \
                and last_paragraph.get("style") == edited_item.get("style"):
            merge_into(last_paragraph, edited_item)
            continue
        edited_items.append(edited_item)
        last_paragraph = edited_item if edited_item["kind"] == "p" else None
    return edited_items
