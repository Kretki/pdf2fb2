"""Units of reference text that look like OCR errors of the reference itself.

Most "correct" corpus books are readable machine OCR (Internet Archive hidden text, NDL OCR), not proofread
text, so their reference chunks still hold OCR errors (``goo4``, ``mo&t``, ``des.^en``, ``〓``). Synthetic
examples must not teach that such units are correct: they are labelled ``ignore`` (kept as context, no loss).
"""

import re
import unicodedata

UNKNOWN_MARKS = ("〓", "�")
LETTER_THEN_DIGIT = re.compile(r"[^\W\d_]\d")
DIGIT_THEN_LETTERS = re.compile(r"\d([^\W\d_]{2,})")
SYMBOL_AT_LETTER = re.compile(r"[\^~|\\{}<>@#%=_][^\W_]|[^\W_][\^~|\\{}<>@#%=_]")
ORDINAL_SUFFIXES = frozenset(("st", "nd", "rd", "th", "er", "re", "ème", "eme", "me", "de", "do", "da", "mo", "ma",
                              "te", "ten", "ter", "го", "ми", "му", "ое", "ая", "ые", "ый", "ий", "ой"))
SYMBOL_INSIDE_WORD = re.compile(r"[^\W_][^\w\s]{0,2}[\^~|\\{}<>@#%&=_*][^\w\s]{0,2}[^\W_]")
EDGE_PUNCTUATION = "\"'«»„“”‘’()[]{}.,;:!?…-—"
CHECKED_SCRIPTS = ("LATIN", "CYRILLIC", "GREEK")


def letter_scripts(word_text: str) -> set[str]:
    """Scripts of the Latin, Cyrillic and Greek letters of a word.

    Args:
        word_text: Unit text.

    Returns:
        Script names found (other scripts are not reported).
    """
    found_scripts = set()
    for word_char in word_text:
        if word_char.isalpha():
            script_name = unicodedata.name(word_char, "").split(" ", 1)[0]
            if script_name in CHECKED_SCRIPTS:
                found_scripts.add(script_name)
    return found_scripts


def is_suspicious_unit(unit_text: str) -> bool:
    """Tell whether a reference unit is probably an OCR error of the reference.

    Usage:
        Called by ``noise_synthesis`` for every reference unit before it is labelled ``correct``.

    Args:
        unit_text: One reference unit.

    Returns:
        True for unknown-character marks, mixed Latin/Cyrillic/Greek letters, a digit right after a letter,
        a digit glued to letters, or a stray symbol between or next to letters.
    """
    if any(unknown_mark in unit_text for unknown_mark in UNKNOWN_MARKS):
        return True
    core_text = unit_text.strip(EDGE_PUNCTUATION)
    return len(letter_scripts(core_text)) > 1 or bool(LETTER_THEN_DIGIT.search(core_text)) \
        or bool(SYMBOL_INSIDE_WORD.search(core_text)) or bool(SYMBOL_AT_LETTER.search(core_text)) \
        or has_glued_digit(core_text)


def has_glued_digit(core_text: str) -> bool:
    """Tell whether a digit is glued to following letters that are not an ordinal suffix (``6hosen``).

    Args:
        core_text: Unit without edge punctuation.

    Returns:
        True for a digit followed by two or more letters other than ``st``, ``th``, ``ème``, ``-го`` and alike.
    """
    return any(letter_run.lower() not in ORDINAL_SUFFIXES for letter_run in DIGIT_THEN_LETTERS.findall(core_text))
