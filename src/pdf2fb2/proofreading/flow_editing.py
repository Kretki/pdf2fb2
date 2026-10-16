"""Apply proofreading edits to the book flow: delete words and blocks, re-join split paragraphs.

Deleting a wrong heading removes the section split it caused (sections are built from headings
when the FB2 is rendered); a paragraph marked as merged is appended to the previous surviving
paragraph, across page markers, undoing end-of-line hyphenation.
"""

from pdf2fb2.fb2_output.text_markup import join_text_fragments
from pdf2fb2.proofreading.edit_decisions import FlowEdits
from pdf2fb2.proofreading.flow_blocks import FlowBlock


def text_without_words(flow_block: FlowBlock, deleted_words: frozenset[tuple[int, int]],
                       block_index: int) -> str | None:
    """Item text with the deleted words removed.

    Args:
        flow_block: Block of the item.
        deleted_words: ``(block_index, word_index)`` pairs.
        block_index: Index of the block.

    Returns:
        The tokens that remain, joined by spaces (``None`` when nothing is deleted).
    """
    removed_tokens = {flow_block.word_tokens[word_index] for word_index in range(len(flow_block.word_tokens))
                      if (block_index, word_index) in deleted_words}
    if not removed_tokens:
        return None
    return " ".join(raw_token for token_index, raw_token in enumerate(flow_block.raw_tokens)
                    if token_index not in removed_tokens)


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
        edited_text = text_without_words(flow_blocks[block_index], flow_edits.deleted_words, block_index)
        edited_item = {**flow_item, "text": flow_item["text"] if edited_text is None else edited_text}
        if block_index in flow_edits.merged_blocks and last_paragraph is not None and edited_item["kind"] == "p" \
                and last_paragraph.get("style") == edited_item.get("style"):
            merge_into(last_paragraph, edited_item)
            continue
        edited_items.append(edited_item)
        last_paragraph = edited_item if edited_item["kind"] == "p" else None
    return edited_items
