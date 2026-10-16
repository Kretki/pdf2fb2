"""Unit tests of the TexTAR word-style application (alignment, markup, FB2 conversion)."""

from pdf2fb2.fb2_output.text_markup import join_text_fragments, markdown_to_fb2_inline
from pdf2fb2.text_styling.style_markup import apply_word_styles, wrap_token
from pdf2fb2.text_styling.token_alignment import align_tokens_to_words, alignable_tokens
from pdf2fb2.text_styling.word_lines import words_in_reading_order


def make_word(word_box: list[int], emphasis_class: int = 0, emphasis_score: float = 0.9) -> dict:
    """Word as returned by the style stage.

    Args:
        word_box: Pixel box.
        emphasis_class: 0 normal, 1 bold, 2 italic, 3 bold+italic.
        emphasis_score: Probability of the class.

    Returns:
        Word dictionary.
    """
    return {"bbox": word_box, "emphasis_class": emphasis_class, "emphasis_score": emphasis_score,
            "line_class": 0, "line_score": 0.9}


def line_of_words(character_counts: list[int], top_edge: int, word_classes: dict[int, int]) -> list[dict]:
    """Boxes of one text line, 10 px per character plus a 10 px gap.

    Args:
        character_counts: Characters per word.
        top_edge: Line top in pixels.
        word_classes: Word index -> emphasis class.

    Returns:
        Words left to right.
    """
    line_words, left_edge = [], 0
    for word_index, character_count in enumerate(character_counts):
        line_words.append(make_word([left_edge, top_edge, left_edge + 10 * character_count, top_edge + 20],
                                    word_classes.get(word_index, 0)))
        left_edge += 10 * character_count + 10
    return line_words


def test_lines_are_read_top_down_left_right() -> None:
    """Words of two lines come back in reading order."""
    shuffled_words = [make_word([100, 40, 150, 60]), make_word([0, 0, 50, 22]), make_word([0, 42, 50, 60]),
                      make_word([100, 2, 150, 20])]
    assert [word_entry["bbox"][:2] for word_entry in words_in_reading_order(shuffled_words)] == \
        [[0, 0], [100, 2], [0, 42], [100, 40]]


def test_alignment_survives_a_merged_box() -> None:
    """Two tokens in one detected box do not shift the following tokens."""
    text_tokens = alignable_tokens("so far entered his soul")
    word_widths = [20, 120, 30, 40]
    assert align_tokens_to_words(text_tokens, word_widths) == [0, 1, 1, 2, 3]


def test_italic_title_is_wrapped_and_merged() -> None:
    """Italic words become one FB2 emphasis element."""
    page_record = {"page": 1, "segments": [{
        "label": "text", "bbox": [0, 0, 400, 40], "text": "read The Robin Hood today",
        "words": line_of_words([4, 3, 5, 4, 5], 0, {1: 2, 2: 2, 3: 2})}]}
    assert apply_word_styles([page_record]) == 1
    styled_text = page_record["segments"][0]["text"]
    assert styled_text == "read <i>The</i> <i>Robin</i> <i>Hood</i> today"
    assert markdown_to_fb2_inline(styled_text) == "read <emphasis>The Robin Hood</emphasis> today"
    assert page_record["segments"][0]["ocr_text"] == "read The Robin Hood today"


def test_fully_styled_heading_is_left_plain() -> None:
    """A heading set entirely in italic keeps its plain text."""
    page_record = {"page": 1, "segments": [{
        "label": "paragraph_title", "bbox": [0, 0, 200, 30], "text": "Chapter One",
        "words": line_of_words([7, 3], 0, {0: 2, 1: 2})}]}
    assert apply_word_styles([page_record]) == 0
    assert page_record["segments"][0]["text"] == "Chapter One"


def test_hyphenated_styled_word_is_rejoined() -> None:
    """A styled word split at a line end is joined back without the hyphen."""
    joined_text = join_text_fragments("<i>exam</i>-", "<i>ple</i> text")
    assert markdown_to_fb2_inline(joined_text) == "<emphasis>example</emphasis> text"


def test_tokens_without_letters_are_not_wrapped() -> None:
    """Footnote markers and numbers keep their form."""
    assert wrap_token("12", ("i",)) == "12"
    assert wrap_token("word*", ("b",)) == "word*"
    assert wrap_token("bold,", ("b", "i")) == "<b><i>bold,</i></b>"
