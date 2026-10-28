"""Misreadings that change one unit, modelled on the genuine OCR errors of the training split.

* confusion substitution (``с``→``е``, ``m``→``rn``, old orthography ``ѣ``→``ъ``) — the most common kind;
* a dropped diacritic (``Théâtre``→``Theatre``, ``á``→``a``);
* a letter or digit deleted, duplicated or swapped.

Every misreading changes a letter or digit: case- and punctuation-only differences are labelled ``ignore`` in
the real data, so they are never produced as ``misread`` here.
"""

import random
import unicodedata

from worker_protocol.ocr_confusions import directed_pairs

# Old-orthography confusions seen in real output; kept here so the proofread worker's pairs stay unchanged.
EXTRA_GROUPS = (("ѣ", "ъ"), ("ѣ", "ь"), ("і", "i"), ("і", "1"), ("й", "и"))
SUBSTITUTION_PAIRS = tuple(sorted(set(directed_pairs()) | {
    directed_pair for first_group, second_group in EXTRA_GROUPS
    for directed_pair in ((first_group, second_group), (second_group, first_group))}))
CONFUSION_SHARE = 0.65
DIACRITIC_SHARE = 0.15


def confusion_misreading(unit_text: str, noise_generator: random.Random) -> str | None:
    """Apply one confusion-pair substitution.

    Args:
        unit_text: Unit to corrupt.
        noise_generator: Random generator.

    Returns:
        The corrupted unit, or None when no pair applies.
    """
    applicable_pairs = [(source_text, target_text) for source_text, target_text in SUBSTITUTION_PAIRS
                        if source_text in unit_text]
    if not applicable_pairs:
        return None
    source_text, target_text = noise_generator.choice(applicable_pairs)
    occurrence_starts = [char_index for char_index in range(len(unit_text))
                         if unit_text.startswith(source_text, char_index)]
    char_index = noise_generator.choice(occurrence_starts)
    return unit_text[:char_index] + target_text + unit_text[char_index + len(source_text):]


def diacritic_misreading(unit_text: str, noise_generator: random.Random) -> str | None:
    """Drop the diacritic of one accented letter.

    Args:
        unit_text: Unit to corrupt.
        noise_generator: Random generator.

    Returns:
        The unit with one accent removed, or None when it has no removable accent.
    """
    accented_positions = [char_index for char_index, word_char in enumerate(unit_text)
                          if unicodedata.normalize("NFD", word_char)[0] != word_char and word_char.isalpha()
                          and unicodedata.normalize("NFD", word_char)[0].isascii()]
    if not accented_positions:
        return None
    char_index = noise_generator.choice(accented_positions)
    return unit_text[:char_index] + unicodedata.normalize("NFD", unit_text[char_index])[0] + \
        unit_text[char_index + 1:]


def character_misreading(unit_text: str, noise_generator: random.Random) -> str:
    """Delete, duplicate or swap a letter or digit of a unit.

    Args:
        unit_text: Unit to corrupt (non-empty).
        noise_generator: Random generator.

    Returns:
        A unit that differs in its letters or digits whenever the unit has any.
    """
    alnum_positions = [char_index for char_index, word_char in enumerate(unit_text) if word_char.isalnum()]
    char_index = noise_generator.choice(alnum_positions or [0])
    edit_kind = noise_generator.choice(("delete", "duplicate", "swap"))
    if edit_kind == "delete" and len(alnum_positions) > 1:
        return unit_text[:char_index] + unit_text[char_index + 1:]
    next_index = char_index + 1
    if edit_kind == "swap" and next_index in alnum_positions and unit_text[char_index] != unit_text[next_index]:
        return unit_text[:char_index] + unit_text[next_index] + unit_text[char_index] + unit_text[next_index + 1:]
    return unit_text[:char_index + 1] + unit_text[char_index] + unit_text[char_index + 1:]


def misread_unit(unit_text: str, noise_generator: random.Random) -> str:
    """Corrupt one unit, preferring a realistic confusion, then a dropped diacritic.

    Usage:
        Called by ``noise_synthesis`` for a unit picked for a single-unit misreading.

    Args:
        unit_text: Unit to corrupt (non-empty).
        noise_generator: Random generator.

    Returns:
        Corrupted unit (differs from the input).
    """
    kind_draw = noise_generator.random()
    corrupted_text = None
    if kind_draw < CONFUSION_SHARE:
        corrupted_text = confusion_misreading(unit_text, noise_generator)
    elif kind_draw < CONFUSION_SHARE + DIACRITIC_SHARE:
        corrupted_text = diacritic_misreading(unit_text, noise_generator)
    return corrupted_text or character_misreading(unit_text, noise_generator)
