"""Edits of a segment text by unit: replace a run of units by a re-read text, delete single units."""

import re
from dataclasses import dataclass

from pdf2fb2.text_verification.verify_units import VerifyUnit

SPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class TextEdit:
    """Replacement of a character range of the segment text.

    Attributes:
        start_offset: First replaced character.
        end_offset: Character after the replaced range.
        new_text: Inserted text (empty to delete).
    """

    start_offset: int
    end_offset: int
    new_text: str


def flat_reading(read_text: str) -> str:
    """A re-read region as one line (the OCR returns Markdown with line breaks).

    Args:
        read_text: OCR text of the region.

    Returns:
        Whitespace-collapsed text.
    """
    return SPACE_PATTERN.sub(" ", read_text).strip()


def deletion_edit(segment_text: str, verify_unit: VerifyUnit) -> TextEdit:
    """Delete one unit together with the whitespace after it.

    Args:
        segment_text: Segment text.
        verify_unit: Unit to delete.

    Returns:
        The edit.
    """
    end_offset = verify_unit.end_offset
    while end_offset < len(segment_text) and segment_text[end_offset] == " ":
        end_offset += 1
    return TextEdit(verify_unit.start_offset, end_offset, "")


def apply_edits(segment_text: str, text_edits: list[TextEdit]) -> str:
    """Apply non-overlapping edits.

    Args:
        segment_text: Original text.
        text_edits: Edits (any order; overlapping later edits are skipped).

    Returns:
        Edited text.
    """
    edited_text, previous_start = segment_text, len(segment_text) + 1
    for text_edit in sorted(text_edits, key=lambda candidate_edit: candidate_edit.start_offset, reverse=True):
        if text_edit.end_offset > previous_start:
            continue
        edited_text = edited_text[:text_edit.start_offset] + text_edit.new_text + edited_text[text_edit.end_offset:]
        previous_start = text_edit.start_offset
    return edited_text.strip()
