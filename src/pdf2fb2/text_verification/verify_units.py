"""Units the token verifier labels: whitespace tokens, with Chinese and Japanese runs cut into characters.

The same split is used for the training data (``benchmark.verifier_export``) and in the pipeline, so the model
always sees the units it was trained on. Every unit keeps its character offsets in the segment text.
"""

import re
from dataclasses import dataclass

UNIT_PATTERN = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿々〆]|"
                          r"[^\s぀-ヿ㐀-䶿一-鿿豈-﫿々〆]+")


@dataclass(frozen=True)
class VerifyUnit:
    """One unit of a segment text.

    Attributes:
        unit_text: Text of the unit.
        start_offset: Offset of its first character in the segment text.
        end_offset: Offset just after it.
    """

    unit_text: str
    start_offset: int
    end_offset: int


def verify_units(segment_text: str) -> list[VerifyUnit]:
    """Split a segment text into verifier units.

    Usage:
        Call on the unstyled OCR text of a segment.

    Args:
        segment_text: Text to split.

    Returns:
        Units in text order.
    """
    return [VerifyUnit(unit_match.group(), unit_match.start(), unit_match.end())
            for unit_match in UNIT_PATTERN.finditer(segment_text)]
