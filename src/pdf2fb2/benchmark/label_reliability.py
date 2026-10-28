"""Which alignment labels are trustworthy enough to train and evaluate the token verifier on.

The labels come from aligning pipeline output with a reference that is itself imperfect (NDL OCR, Internet
Archive hidden text, other editions). Three kinds of label are mostly reference noise, not OCR errors:

* a page whose output barely matches its located reference (wrong or poor reference: everything looks
  ``inserted`` or ``misread``) — the whole page is dropped;
* a segment that is mostly ``inserted`` (a block the reference has elsewhere or lacks) — the segment is dropped;
* a ``misread`` pair that differs only in case, punctuation or footnote markers (``Пролет.`` / ``Пролет``,
  ``ассоциациям¹)`` / ``ассоциациям``) — the unit is labelled ``ignore`` (no loss, no metric).

Two exceptions keep genuine pipeline errors in the data: pages and segments with a repetition loop (the OCR
model ran away: a real ``inserted`` error, not a reference problem) and every page of a hand-proofread reference
(its labels are exact; a low match there means bad OCR, which the verifier must learn).
"""

import unicodedata

from pdf2fb2.benchmark.text_normalization import match_key

MIN_PAGE_CORRECT_SHARE = 0.35    # pages below this share of correct units have an unusable reference
MAX_SEGMENT_INSERTED_SHARE = 0.5  # segments at or above this share of inserted units are alignment failures
MIN_JUDGED_UNITS = 4              # shorter segments are too small to judge and are kept
JUDGED_LABELS = ("correct", "misread", "inserted")
LOOP_REPEATS = 3                  # an n-gram repeated this often in a row is a repetition loop
MAX_LOOP_UNITS = 8                # longest repeated n-gram looked for


def lenient_key(word_text: str) -> str:
    """Word reduced to letters and decimal digits (case, punctuation and superscript markers dropped).

    Args:
        word_text: Normalised word.

    Returns:
        Lower-case letters and decimal digits of the word.
    """
    return "".join(word_char for word_char in match_key(word_text) if unicodedata.category(word_char) != "No")


def is_convention_difference(output_word: str, reference_word: str) -> bool:
    """Tell whether a misread pair differs only by case, punctuation or footnote markers.

    Args:
        output_word: Normalised output word.
        reference_word: Normalised reference word it was aligned with.

    Returns:
        True when both words have the same non-empty lenient key.
    """
    output_key = lenient_key(output_word)
    return bool(output_key) and output_key == lenient_key(reference_word)


def label_share(unit_labels: list[str], label_name: str) -> tuple[float, int]:
    """Share of one label among the judged units.

    Args:
        unit_labels: Labels of a page or segment (``ignore`` units are not judged).
        label_name: Label to count.

    Returns:
        ``(share 0..1, number of judged units)``; the share is 0 when nothing is judged.
    """
    judged_labels = [unit_label for unit_label in unit_labels if unit_label in JUDGED_LABELS]
    return judged_labels.count(label_name) / max(len(judged_labels), 1), len(judged_labels)


def has_repetition_loop(unit_texts: list[str]) -> bool:
    """Tell whether some run of 1..``MAX_LOOP_UNITS`` units repeats ``LOOP_REPEATS`` times in a row.

    Args:
        unit_texts: Units in reading order.

    Returns:
        True when a repetition loop is present.
    """
    for gram_size in range(1, MAX_LOOP_UNITS + 1):
        for start_index in range(len(unit_texts) - gram_size * LOOP_REPEATS + 1):
            unit_gram = unit_texts[start_index:start_index + gram_size]
            if all(unit_texts[start_index + repeat_index * gram_size:start_index + (repeat_index + 1) * gram_size]
                   == unit_gram for repeat_index in range(1, LOOP_REPEATS)):
                return True
    return False


def page_is_reliable(page_labels: list[str], proofread_reference: bool = False,
                     page_units: list[str] | None = None) -> bool:
    """Tell whether a page's labels can be trusted.

    Usage:
        Called by ``verifier_export`` once per located page.

    Args:
        page_labels: Labels of all units of the page (page-level alignment).
        proofread_reference: The book's reference is hand-proofread (always trusted).
        page_units: Units of the page (a repetition loop keeps the page).

    Returns:
        True for proofread references, pages with a repetition loop, or pages where at least
        ``MIN_PAGE_CORRECT_SHARE`` of the judged units are correct.
    """
    if proofread_reference or (page_units and has_repetition_loop(page_units)):
        return True
    correct_share, judged_count = label_share(page_labels, "correct")
    return judged_count > 0 and correct_share >= MIN_PAGE_CORRECT_SHARE


def segment_is_reliable(segment_labels: list[str], proofread_reference: bool = False,
                        segment_units: list[str] | None = None) -> bool:
    """Tell whether a segment's labels are not an alignment failure.

    Usage:
        Called by ``verifier_export`` for every segment of a reliable page.

    Args:
        segment_labels: Labels of the segment's units.
        proofread_reference: The book's reference is hand-proofread (always trusted).
        segment_units: Units of the segment (a repetition loop keeps it).

    Returns:
        False when a segment of at least ``MIN_JUDGED_UNITS`` judged units is mostly ``inserted`` without a
        repetition loop and without a proofread reference.
    """
    if proofread_reference or (segment_units and has_repetition_loop(segment_units)):
        return True
    inserted_share, judged_count = label_share(segment_labels, "inserted")
    return judged_count < MIN_JUDGED_UNITS or inserted_share < MAX_SEGMENT_INSERTED_SHARE
