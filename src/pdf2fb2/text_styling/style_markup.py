"""Insert ``<b>``/``<i>``/``<s>`` tags into OCR text from the word attributes predicted by TexTAR.

The tags are the inline HTML the FB2 converter already understands (``text_markup``): ``<b>`` ->
``<strong>``, ``<i>`` -> ``<emphasis>``, ``<s>`` -> ``<strikethrough>``. Every token is wrapped on
its own (adjacent equal tags are merged during FB2 conversion), so no tag ever spans a line or a
paragraph break. FB2 has no underline element, so underlines are not applied.
"""

import re

from pdf2fb2.segments.label_roles import UNIFORM_STYLE_DROPPED_LABELS, receives_word_styles
from pdf2fb2.text_styling.style_consensus import consensus_words, italic_dominates
from pdf2fb2.text_styling.token_alignment import TextToken, align_tokens_to_words, alignable_tokens
from pdf2fb2.text_styling.word_lines import words_in_reading_order

MIN_STYLE_SCORE = 0.4
MIN_BOLD_SCORE = 0.55  # weaker bold hits on 212 test pages were all regular words
MIN_STRIKEOUT_SCORE = 0.7  # printed strikeout is rare; weaker hits are mostly reader marks
BOLD_CLASSES = frozenset({1, 3})
ITALIC_CLASSES = frozenset({2, 3})
STRIKEOUT_CLASSES = frozenset({2, 3})
LETTER_PATTERN = re.compile(r"[^\W\d_]")
MARKUP_CHARACTER_PATTERN = re.compile(r"[<>$&*_~`\[\]|\\]")
HYPHEN_CHARACTERS = "-­"


def word_style_tags(word_entry: dict, minimum_score: float) -> tuple[str, ...]:
    """Tags (outermost first) for one detected word.

    Args:
        word_entry: Word from the style stage (classes and scores).
        minimum_score: Lowest averaged probability for a style to be applied.

    Returns:
        Subset of ``("s", "b", "i")`` in this order.
    """
    emphasis_class = word_entry["emphasis_class"]
    emphasis_score = word_entry["emphasis_score"]
    line_class = word_entry["line_class"] if word_entry["line_score"] >= max(minimum_score, MIN_STRIKEOUT_SCORE) else 0
    italic_active = emphasis_class in ITALIC_CLASSES and emphasis_score >= minimum_score
    bold_active = emphasis_class in BOLD_CLASSES and emphasis_score >= max(minimum_score, MIN_BOLD_SCORE)
    return tuple(tag_name for tag_name, is_active in (("s", line_class in STRIKEOUT_CLASSES), ("b", bold_active),
                                                      ("i", italic_active)) if is_active)


def wrap_token(token_text: str, tag_names: tuple[str, ...]) -> str:
    """Wrap a token in style tags; a trailing hyphen stays outside so hyphenation can be undone.

    Tokens without letters (footnote markers, numbers) or with Markdown/HTML characters are left
    unchanged.

    Args:
        token_text: Token from the OCR text.
        tag_names: Tags from ``word_style_tags``.

    Returns:
        The (possibly) wrapped token.
    """
    if not tag_names or not LETTER_PATTERN.search(token_text) or MARKUP_CHARACTER_PATTERN.search(token_text):
        return token_text
    core_text, hyphen_suffix = token_text, ""
    if len(token_text) > 1 and token_text[-1] in HYPHEN_CHARACTERS:
        core_text, hyphen_suffix = token_text[:-1], token_text[-1]
    opening_tags = "".join(f"<{tag_name}>" for tag_name in tag_names)
    closing_tags = "".join(f"</{tag_name}>" for tag_name in reversed(tag_names))
    return f"{opening_tags}{core_text}{closing_tags}{hyphen_suffix}"


def rebuild_text(ocr_text: str, text_tokens: list[TextToken], token_tags: list[tuple[str, ...]]) -> str:
    """Replace every token by its wrapped form, keeping all whitespace.

    Args:
        ocr_text: Original text.
        text_tokens: Tokens of ``ocr_text`` in order.
        token_tags: Tags of every token.

    Returns:
        Styled text.
    """
    text_pieces = []
    previous_end = 0
    for text_token, tag_names in zip(text_tokens, token_tags, strict=True):
        text_pieces.append(ocr_text[previous_end:text_token.start_offset])
        text_pieces.append(wrap_token(ocr_text[text_token.start_offset:text_token.end_offset], tag_names))
        previous_end = text_token.end_offset
    text_pieces.append(ocr_text[previous_end:])
    return "".join(text_pieces)


def styled_segment_text(page_segment: dict, minimum_score: float = MIN_STYLE_SCORE, italic_page: bool = False) -> str:
    """OCR text of a segment with the TexTAR styles applied.

    Args:
        page_segment: Segment with ``text``, ``label`` and ``words`` (from the style stage).
        minimum_score: Lowest probability for a style to be applied.
        italic_page: Whether bold+italic is read as italic on this page (``italic_dominates``).

    Returns:
        Styled text; the unchanged text when nothing can be aligned or applied.
    """
    ocr_text = page_segment.get("text") or ""
    ordered_words = words_in_reading_order(consensus_words(page_segment.get("words") or [], italic_page))
    text_tokens = alignable_tokens(ocr_text)
    word_indexes = align_tokens_to_words(text_tokens, [word_entry["bbox"][2] - word_entry["bbox"][0]
                                                       for word_entry in ordered_words])
    if word_indexes is None:
        return ocr_text
    token_tags = [word_style_tags(ordered_words[word_index], minimum_score) if word_index is not None else ()
                  for word_index in word_indexes]
    if page_segment["label"] in UNIFORM_STYLE_DROPPED_LABELS and all(token_tags):
        return ocr_text
    return rebuild_text(ocr_text, text_tokens, token_tags)


def apply_word_styles(page_records: list[dict], minimum_score: float = MIN_STYLE_SCORE) -> int:
    """Style the text of every segment that has detected words.

    Usage:
        Call after OCR and the style stage, before FB2 assembly. The unstyled OCR text is kept in
        ``ocr_text``.

    Args:
        page_records: OCR'd page records (mutated).
        minimum_score: Lowest probability for a style to be applied.

    Returns:
        Number of segments whose text changed.
    """
    changed_segments = 0
    for page_record in page_records:
        styled_segments = [page_segment for page_segment in page_record["segments"]
                           if receives_word_styles(page_segment["label"]) and page_segment.get("words")]
        italic_page = italic_dominates([word_entry for page_segment in styled_segments
                                        for word_entry in page_segment["words"]], minimum_score)
        for page_segment in styled_segments:
            styled_text = styled_segment_text(page_segment, minimum_score, italic_page)
            page_segment["ocr_text"] = page_segment.get("text") or ""
            if styled_text != page_segment["ocr_text"]:
                page_segment["text"] = styled_text
                changed_segments += 1
    return changed_segments
