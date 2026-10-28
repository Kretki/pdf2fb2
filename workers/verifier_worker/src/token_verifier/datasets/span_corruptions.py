"""Misreadings that change the unit boundaries, labelled the way the real alignment labels them.

* merged units: a missing space joins two units (``hombre,ó``, ``очередь—земельному``) — one ``misread`` unit;
* split unit: an extra space or an unjoined line-break hyphen cuts a word (``commu- nity``) — the first piece is
  ``misread``, the second ``inserted`` (the reference has one word for both).
"""

import random
import unicodedata
from dataclasses import dataclass

MIN_SPLIT_LETTERS = 6
CJK_NAME_PREFIXES = ("CJK", "HIRAGANA", "KATAKANA")


@dataclass(frozen=True)
class CorruptionResult:
    """Output of one corruption.

    Attributes:
        labelled_units: ``(unit text, label)`` pairs that replace the reference units.
        consumed_units: Reference units replaced (1 or 2).
    """

    labelled_units: tuple[tuple[str, str], ...]
    consumed_units: int


def is_cjk_unit(unit_text: str) -> bool:
    """Tell whether a unit is a Han or kana character (CJK units are never merged or split).

    Args:
        unit_text: Unit text.

    Returns:
        True when the unit contains a CJK ideograph or kana.
    """
    return any(unicodedata.name(word_char, "").startswith(CJK_NAME_PREFIXES) for word_char in unit_text)


def merged_units(first_unit: str, second_unit: str) -> CorruptionResult | None:
    """Join two units as if the space between them was lost.

    Args:
        first_unit: Unit at the corrupted position.
        second_unit: Following reference unit (empty at the end of a segment).

    Returns:
        One ``misread`` unit replacing both, or None when merging does not apply.
    """
    if not second_unit or is_cjk_unit(first_unit) or is_cjk_unit(second_unit) \
            or not any(word_char.isalnum() for word_char in first_unit + second_unit):
        return None
    return CorruptionResult(((first_unit + second_unit, "misread"),), 2)


def split_unit(unit_text: str, noise_generator: random.Random) -> CorruptionResult | None:
    """Cut a long word in two, with or without a leftover line-break hyphen.

    Args:
        unit_text: Unit at the corrupted position.
        noise_generator: Random generator.

    Returns:
        ``misread`` first piece and ``inserted`` second piece, or None for short or CJK units.
    """
    letter_positions = [char_index for char_index, word_char in enumerate(unit_text) if word_char.isalpha()]
    if len(letter_positions) < MIN_SPLIT_LETTERS or is_cjk_unit(unit_text):
        return None
    cut_index = noise_generator.choice(letter_positions[2:-2])
    hyphen_text = "-" if noise_generator.random() < 0.5 else ""
    return CorruptionResult(((unit_text[:cut_index] + hyphen_text, "misread"), (unit_text[cut_index:], "inserted")),
                            1)
