"""Segment-level clean-up of TexTAR word predictions before they are aligned with the OCR text.

* Boxes much smaller than the segment's typical word (dots, specks) are dropped.
* Bold+italic is rare; when the italic-only words of a page clearly outnumber its bold+italic
  words, the latter are treated as italic (TexTAR tends to add "bold" to some italic words).
"""

ITALIC_CLASS = 2
BOLD_ITALIC_CLASS = 3
MIN_HEIGHT_RATIO = 0.5
ITALIC_MAJORITY_RATIO = 3.0


def median_height(segment_words: list[dict]) -> float:
    """Median box height of the words of a segment.

    Args:
        segment_words: Words with ``bbox``.

    Returns:
        Height in pixels (0 for no words).
    """
    word_heights = sorted(word_entry["bbox"][3] - word_entry["bbox"][1] for word_entry in segment_words)
    return float(word_heights[len(word_heights) // 2]) if word_heights else 0.0


def drop_specks(segment_words: list[dict]) -> list[dict]:
    """Remove boxes lower than half the median word height.

    Args:
        segment_words: Words of one segment.

    Returns:
        Remaining words in the same order.
    """
    minimum_height = MIN_HEIGHT_RATIO * median_height(segment_words)
    return [word_entry for word_entry in segment_words
            if word_entry["bbox"][3] - word_entry["bbox"][1] >= minimum_height]


def italic_dominates(page_words: list[dict], minimum_score: float) -> bool:
    """Tell whether bold+italic predictions of a page should be read as italic.

    Args:
        page_words: Words of all styled segments of one page.
        minimum_score: Lowest score of a counted prediction.

    Returns:
        True when confident italic-only words outnumber bold+italic ones at least 3:1.
    """
    confident_classes = [word_entry["emphasis_class"] for word_entry in page_words
                         if word_entry["emphasis_score"] >= minimum_score]
    bold_italic_count = confident_classes.count(BOLD_ITALIC_CLASS)
    italic_count = confident_classes.count(ITALIC_CLASS)
    return bold_italic_count > 0 and italic_count >= ITALIC_MAJORITY_RATIO * bold_italic_count


def consensus_words(segment_words: list[dict], italic_page: bool) -> list[dict]:
    """Apply the clean-up rules to the words of a segment.

    Usage:
        Call before ordering and aligning the words with the OCR tokens.

    Args:
        segment_words: Words from the style stage.
        italic_page: Result of ``italic_dominates`` for the segment's page.

    Returns:
        Cleaned words (copies where a class changed).
    """
    kept_words = drop_specks(segment_words)
    if not italic_page:
        return kept_words
    return [{**word_entry, "emphasis_class": ITALIC_CLASS} if word_entry["emphasis_class"] == BOLD_ITALIC_CLASS
            else word_entry for word_entry in kept_words]
