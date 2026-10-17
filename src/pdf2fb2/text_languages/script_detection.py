"""Writing script of characters and texts, from Unicode code-point ranges (no model, no data files).

Scripts are what can be told from the characters alone: Latin, Cyrillic, Greek, Han, kana, Hangul, Arabic,
Hebrew, Devanagari and Thai. Which language of a script a text is in is decided by ``language_detection``.
Han, kana and Thai are written without spaces between words, so line joins must not insert one.
"""

import bisect
from collections import Counter

LATIN_SCRIPT = "latin"
CYRILLIC_SCRIPT = "cyrillic"
GREEK_SCRIPT = "greek"
HAN_SCRIPT = "han"
KANA_SCRIPT = "kana"
HANGUL_SCRIPT = "hangul"
ARABIC_SCRIPT = "arabic"
HEBREW_SCRIPT = "hebrew"
DEVANAGARI_SCRIPT = "devanagari"
THAI_SCRIPT = "thai"
CJK_PUNCTUATION = "cjk_punctuation"  # 。、「」 and full-width forms: no script of their own, but spaceless

# (first code point, last code point, script), sorted by the first code point
SCRIPT_RANGES = (
    (0x0041, 0x005A, LATIN_SCRIPT), (0x0061, 0x007A, LATIN_SCRIPT), (0x00C0, 0x024F, LATIN_SCRIPT),
    (0x0370, 0x03FF, GREEK_SCRIPT), (0x0400, 0x052F, CYRILLIC_SCRIPT), (0x0590, 0x05FF, HEBREW_SCRIPT),
    (0x0600, 0x06FF, ARABIC_SCRIPT), (0x0900, 0x097F, DEVANAGARI_SCRIPT), (0x0E00, 0x0E7F, THAI_SCRIPT),
    (0x1100, 0x11FF, HANGUL_SCRIPT), (0x1E00, 0x1EFF, LATIN_SCRIPT), (0x1F00, 0x1FFF, GREEK_SCRIPT),
    (0x3000, 0x303F, CJK_PUNCTUATION), (0x3040, 0x30FF, KANA_SCRIPT), (0x3130, 0x318F, HANGUL_SCRIPT),
    (0x31F0, 0x31FF, KANA_SCRIPT), (0x3400, 0x4DBF, HAN_SCRIPT), (0x4E00, 0x9FFF, HAN_SCRIPT),
    (0xAC00, 0xD7AF, HANGUL_SCRIPT), (0xF900, 0xFAFF, HAN_SCRIPT), (0xFF00, 0xFF60, CJK_PUNCTUATION),
    (0xFF66, 0xFF9F, KANA_SCRIPT), (0x20000, 0x2FA1F, HAN_SCRIPT),
)
RANGE_STARTS = tuple(script_range[0] for script_range in SCRIPT_RANGES)
SPACELESS_SCRIPTS = frozenset({HAN_SCRIPT, KANA_SCRIPT, THAI_SCRIPT, CJK_PUNCTUATION})
SPACELESS_LETTER_WEIGHT = 3  # a Han/kana/Thai character carries about as much text as three alphabet letters


def character_script(text_character: str) -> str | None:
    """Script of one character.

    Args:
        text_character: A single character.

    Returns:
        One of the ``*_SCRIPT`` names or ``CJK_PUNCTUATION``; None for digits, spaces, ASCII punctuation and
        characters of other scripts.
    """
    code_point = ord(text_character)
    range_index = bisect.bisect_right(RANGE_STARTS, code_point) - 1
    if range_index < 0:
        return None
    first_point, last_point, script_name = SCRIPT_RANGES[range_index]
    return script_name if first_point <= code_point <= last_point else None


def count_scripts(plain_text: str) -> Counter:
    """Measure how much of a text is written in each script.

    Usage:
        Call on text without markup (tags would count as Latin letters).

    Args:
        plain_text: Any text.

    Returns:
        Counter script name -> weighted letter count: one per alphabet letter, ``SPACELESS_LETTER_WEIGHT`` per
        Han, kana or Thai character (so a Japanese sentence is not outweighed by a shorter English quotation);
        punctuation (including CJK punctuation) is not counted.
    """
    script_counts = Counter(character_script(text_character) for text_character in plain_text)
    script_counts.pop(None, None)
    script_counts.pop(CJK_PUNCTUATION, None)
    for script_name in SPACELESS_SCRIPTS & script_counts.keys():
        script_counts[script_name] *= SPACELESS_LETTER_WEIGHT
    return script_counts


def is_spaceless_character(text_character: str) -> bool:
    """Tell whether a character belongs to a script written without spaces between words.

    Args:
        text_character: A single character (an empty string gives False).

    Returns:
        True for Han, kana, Thai and CJK punctuation.
    """
    return bool(text_character) and character_script(text_character) in SPACELESS_SCRIPTS
