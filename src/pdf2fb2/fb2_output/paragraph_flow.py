"""Paragraph-level flow helpers: printed page numbers and paragraph re-joining."""

import re

from pdf2fb2.fb2_output.text_markup import SENTENCE_END_PATTERN, join_text_fragments
from pdf2fb2.segments.label_roles import PAGE_NUMBER_LABELS

DIGITS_PATTERN = re.compile(r"\d+")


def find_printed_page_number(page_record: dict) -> str | None:
    """Return the page number printed on the page, if the layout model found one.

    Args:
        page_record: Page record with OCR'd segments.

    Returns:
        Digits of the last ``number`` segment, or None.
    """
    printed_number = None
    for page_segment in page_record["segments"]:
        if page_segment["label"] in PAGE_NUMBER_LABELS:
            digits_match = DIGITS_PATTERN.search(page_segment.get("text", ""))
            printed_number = digits_match.group() if digits_match else printed_number
    return printed_number


def _continues_previous(previous_item: dict | None, paragraph_text: str, paragraph_style: str,
                        is_list_item: bool) -> bool:
    """Tell whether a paragraph continues the previous one across a page/column break.

    Args:
        previous_item: Last non-page flow item, or None.
        paragraph_text: New paragraph text.
        paragraph_style: ``p`` or ``cite``.
        is_list_item: Whether the new paragraph is a list item.

    Returns:
        True if the previous paragraph has the same style, is not a list item, does not
        end a sentence and the new one starts in lowercase.
    """
    return (previous_item is not None and previous_item["kind"] == "p" and previous_item["style"] == paragraph_style
            and not previous_item["list"] and not is_list_item and paragraph_text[:1].islower()
            and not SENTENCE_END_PATTERN.search(previous_item["text"]))


def merge_or_append_paragraph(flow_items: list[dict], paragraph_text: str, page_number: int,
                              paragraph_style: str, is_list_item: bool) -> None:
    """Append a paragraph, or join it to the previous one when it is a continuation.

    Args:
        flow_items: Flow being built (mutated).
        paragraph_text: Paragraph text (Markdown inline).
        page_number: Page the paragraph comes from.
        paragraph_style: ``p`` or ``cite``.
        is_list_item: Whether the paragraph is a list item.

    Returns:
        None.
    """
    previous_item = next((flow_item for flow_item in reversed(flow_items) if flow_item["kind"] != "page"), None)
    if _continues_previous(previous_item, paragraph_text, paragraph_style, is_list_item):
        previous_item["text"] = join_text_fragments(previous_item["text"], paragraph_text)
        previous_item["pages"].add(page_number)
        return
    flow_items.append({"kind": "p", "text": paragraph_text, "style": paragraph_style, "list": is_list_item,
                       "pages": {page_number}})
