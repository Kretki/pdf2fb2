"""Synthetic training examples: reference text of correct books corrupted the way OCR corrupts it.

* misread units (label ``misread``): an OCR confusion substitution (``с``→``е``, ``m``→``rn``, long ``s``→``f``)
  when the unit has one, otherwise a random character deletion, duplication or swap;
* inserted spans (label ``inserted``): units that are not on the page, of three kinds seen in real output —
  a repetition loop of the preceding units (generation running away), a passage of the same book from elsewhere
  (plausible context the reader model "remembered") and short reader notes (numbers, dates, ticks).

Every other unit keeps the label ``correct``.
"""

import random

from token_verifier.datasets.example_files import TokenExample
from token_verifier.verifier_settings import SynthesisSettings
from worker_protocol.ocr_confusions import directed_pairs

SUBSTITUTION_PAIRS = directed_pairs()
NOTE_TEMPLATES = ("{number}", "{number}.", "см. {number}", "p. {number}", "{number}!", "NB", "✓", "×", "?!",
                  "{year} г.",
                  "{year}", "ib.", "S. {number}")


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


def character_misreading(unit_text: str, noise_generator: random.Random) -> str:
    """Delete, duplicate or swap characters of a unit.

    Args:
        unit_text: Unit to corrupt (non-empty).
        noise_generator: Random generator.

    Returns:
        A different unit when possible.
    """
    char_index = noise_generator.randrange(len(unit_text))
    edit_kind = noise_generator.choice(("delete", "duplicate", "swap"))
    if edit_kind == "delete" and len(unit_text) > 1:
        return unit_text[:char_index] + unit_text[char_index + 1:]
    if edit_kind == "swap" and char_index + 1 < len(unit_text) and unit_text[char_index] != unit_text[char_index + 1]:
        return unit_text[:char_index] + unit_text[char_index + 1] + unit_text[char_index] + unit_text[char_index + 2:]
    return unit_text[:char_index + 1] + unit_text[char_index] + unit_text[char_index + 1:]


def misread_unit(unit_text: str, noise_generator: random.Random) -> str:
    """Corrupt one unit, preferring a realistic confusion.

    Args:
        unit_text: Unit to corrupt.
        noise_generator: Random generator.

    Returns:
        Corrupted unit (differs from the input).
    """
    corrupted_text = confusion_misreading(unit_text, noise_generator) if noise_generator.random() < 0.7 else None
    return corrupted_text or character_misreading(unit_text, noise_generator)


def reader_note(noise_generator: random.Random) -> list[str]:
    """Units of a short handwritten-style note.

    Args:
        noise_generator: Random generator.

    Returns:
        One to three units.
    """
    note_text = noise_generator.choice(NOTE_TEMPLATES).format(number=noise_generator.randint(1, 999),
                                                               year=noise_generator.randint(1700, 1950))
    return note_text.split()


def inserted_span(unit_texts: list[str], insert_position: int, donor_units: list[str],
                  synthesis_settings: SynthesisSettings, noise_generator: random.Random) -> list[str]:
    """Units of one inserted span.

    Args:
        unit_texts: Units of the example before the insertion.
        insert_position: Where the span goes.
        donor_units: Units of another passage of the same book (may be empty).
        synthesis_settings: Span length limit.
        noise_generator: Random generator.

    Returns:
        Non-empty list of units.
    """
    span_length = noise_generator.randint(1, synthesis_settings.max_insert_units)
    span_kind = noise_generator.choice(("repeat", "donor", "note"))
    if span_kind == "repeat" and insert_position >= 2:
        return unit_texts[max(0, insert_position - span_length):insert_position]
    if span_kind == "donor" and len(donor_units) > span_length:
        donor_start = noise_generator.randrange(len(donor_units) - span_length)
        return donor_units[donor_start:donor_start + span_length]
    return reader_note(noise_generator)


def corrupt_chunk(book_key: str, language_name: str, reference_units: list[str], donor_units: list[str],
                  synthesis_settings: SynthesisSettings, noise_generator: random.Random) -> TokenExample:
    """Build one synthetic example from a reference chunk.

    Usage:
        Called by ``synthesis_service`` several times per chunk with different random states.

    Args:
        book_key: Source book.
        language_name: Corpus language.
        reference_units: Units of the chunk (taken as correct).
        donor_units: Units of another chunk of the same book.
        synthesis_settings: Rates and limits.
        noise_generator: Random generator.

    Returns:
        The labelled example.
    """
    unit_texts, unit_labels = [], []
    for unit_text in reference_units:
        if unit_text and noise_generator.random() < synthesis_settings.misread_rate:
            unit_texts.append(misread_unit(unit_text, noise_generator))
            unit_labels.append("misread")
        else:
            unit_texts.append(unit_text)
            unit_labels.append("correct")
    if unit_texts and noise_generator.random() < synthesis_settings.insert_rate:
        insert_position = noise_generator.randrange(len(unit_texts) + 1)
        span_units = inserted_span(unit_texts, insert_position, donor_units, synthesis_settings, noise_generator)
        unit_texts[insert_position:insert_position] = span_units
        unit_labels[insert_position:insert_position] = ["inserted"] * len(span_units)
    return TokenExample(book_key, language_name, tuple(unit_texts), tuple(unit_labels))
