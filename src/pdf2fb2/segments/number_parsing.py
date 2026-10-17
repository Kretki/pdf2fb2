"""Reading printed page numbers from OCR text: arabic, roman and prefixed numbering (``A-12``, ``S.3``).

OCR often confuses digits with letters (``IIO`` for 110, ``x11`` for xii, ``$5^{8}$`` for 58), so a text gives
every plausible reading; the numbering check (``segments.page_numbering``) keeps the one that continues the
sequence of the neighbouring pages.
"""

import re
import unicodedata
from dataclasses import dataclass

ARABIC_KIND = "arabic"
ROMAN_KIND = "roman"
MAXIMUM_NUMBER_LENGTH = 12  # characters of a cleaned page-number text
MAXIMUM_ARABIC_DIGITS = 4
MARKUP_PATTERN = re.compile(r"</?\w+>|\\[A-Za-z]+|[$^{}*_`~]")  # HTML tags, LaTeX commands, Markdown marks
INNER_SPACE_PATTERN = re.compile(r"\s+")
DECORATION_CHARACTERS = " \t\n.,:;-–—|[]()<>\"'·•"
PREFIX_PATTERN = re.compile(r"^([A-Za-z]{1,3}[-.–:])(.+)$")
ROMAN_PATTERN = re.compile(r"^m{0,3}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$")
ROMAN_VALUES = {"m": 1000, "d": 500, "c": 100, "l": 50, "x": 10, "v": 5, "i": 1}
# OCR confusions: letters read for digits, and digits read for roman letters.
ARABIC_LOOKALIKES = str.maketrans({"I": "1", "l": "1", "i": "1", "|": "1", "O": "0", "o": "0", "S": "5",
                                   "B": "8", "Z": "2"})
ROMAN_LOOKALIKES = str.maketrans({"1": "i"})


@dataclass(frozen=True)
class NumberReading:
    """One plausible value of a printed page number.

    Attributes:
        numbering_key: (kind, prefix) of the numbering system, e.g. ("roman", "") or ("arabic", "a-").
        number_value: The value (>= 1).
    """

    numbering_key: tuple[str, str]
    number_value: int


def clean_number_text(ocr_text: str) -> str:
    """Strip markup and decorations around a page-number text.

    Args:
        ocr_text: Text of the segment as read by the OCR model.

    Returns:
        The core text without inner spaces, e.g. ``58`` for ``$5^{8}$``, ``41`` for ``$ 4 ^ { \\mathrm { I } } $``
        and ``xii`` for ``— xii —``.
    """
    return INNER_SPACE_PATTERN.sub("", MARKUP_PATTERN.sub("", ocr_text)).strip(DECORATION_CHARACTERS)


def roman_value(roman_text: str) -> int | None:
    """Value of a lowercase roman numeral.

    Args:
        roman_text: Candidate numeral in lowercase.

    Returns:
        Its value, or None when it is not a well-formed numeral.
    """
    if not roman_text or not ROMAN_PATTERN.match(roman_text):
        return None
    total_value = 0
    for letter_index, roman_letter in enumerate(roman_text):
        letter_value = ROMAN_VALUES[roman_letter]
        following_value = ROMAN_VALUES[roman_text[letter_index + 1]] if letter_index + 1 < len(roman_text) else 0
        total_value += -letter_value if letter_value < following_value else letter_value
    return total_value


def unicode_numeral_value(core_text: str) -> int | None:
    """Value of a single Unicode number character such as ``Ⅻ``.

    Args:
        core_text: Cleaned text.

    Returns:
        Its integer value, or None when the text is not one non-ASCII numeric character with an integer value.
    """
    if len(core_text) != 1 or core_text.isascii():
        return None
    numeric_value = unicodedata.numeric(core_text, -1.0)
    return int(numeric_value) if numeric_value >= 1 and numeric_value == int(numeric_value) else None


def core_readings(core_text: str, prefix_text: str) -> list[NumberReading]:
    """Arabic and roman readings of a number core.

    Args:
        core_text: Number text without prefix and decorations.
        prefix_text: Lowercase prefix of the numbering (may be empty).

    Returns:
        Readings, arabic first; empty when the core is not a number.
    """
    number_readings = []
    arabic_text = core_text.translate(ARABIC_LOOKALIKES)
    if arabic_text.isdigit() and len(arabic_text) <= MAXIMUM_ARABIC_DIGITS and int(arabic_text) > 0:
        number_readings.append(NumberReading((ARABIC_KIND, prefix_text), int(arabic_text)))
    numeral_value = roman_value(core_text.lower().translate(ROMAN_LOOKALIKES)) or unicode_numeral_value(core_text)
    if numeral_value is not None:
        number_readings.append(NumberReading((ROMAN_KIND, prefix_text), numeral_value))
    return number_readings


def read_page_number(ocr_text: str) -> list[NumberReading]:
    """Every plausible reading of a printed page number.

    Usage:
        Call on the OCR text of every page-number candidate.

    Args:
        ocr_text: Text of the segment.

    Returns:
        Readings without prefix, plus readings with a letter prefix such as ``A-``; empty for texts longer
        than ``MAXIMUM_NUMBER_LENGTH`` or without a number.
    """
    core_text = clean_number_text(ocr_text)
    if not core_text or len(core_text) > MAXIMUM_NUMBER_LENGTH:
        return []
    number_readings = core_readings(core_text, "")
    prefix_match = PREFIX_PATTERN.match(core_text)
    if prefix_match is not None:
        number_readings += core_readings(prefix_match.group(2).strip(DECORATION_CHARACTERS),
                                         prefix_match.group(1).lower())
    return number_readings
