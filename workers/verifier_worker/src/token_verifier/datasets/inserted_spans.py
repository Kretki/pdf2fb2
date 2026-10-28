"""Inserted spans: units that are not on the page, in the proportions seen in real output of the training split.

Real insertions are mostly one unit long: footnote markers (``¹``, ``1)``, the formula form ``$ ^{1} $``),
stray letters and marks (``g``, ``|``, ``---``), short reader notes, and occasionally a repeated or remembered
run of words. The texts are generated from templates and from the example's own book, never copied from
pipeline output.
"""

import random

from token_verifier.verifier_settings import SynthesisSettings

FOOTNOTE_MARKERS = ("¹", "²", "³", "*", "**", "1)", "2)", "3)", "[1]", "$ ^{1} $", "$ ^{2} $", "^{1}")
STRAY_MARKS = ("|", "-", "---", "'", "“", ".", ",", "•", "~", "l", "i", "I")
NOTE_TEMPLATES = ("{number}", "{number}.", "см. {number}", "p. {number}", "{number}!", "NB", "✓", "×", "?!",
                  "{year} г.", "{year}", "ib.", "S. {number}")
SPAN_KINDS = ("footnote", "stray", "letter", "note", "repeat", "donor")
SPAN_WEIGHTS = (0.3, 0.2, 0.1, 0.15, 0.1, 0.15)


def run_length(synthesis_settings: SynthesisSettings, noise_generator: random.Random) -> int:
    """Length of a repeated or remembered run: geometric, mostly one or two units.

    Args:
        synthesis_settings: ``max_insert_units``.
        noise_generator: Random generator.

    Returns:
        1 .. ``max_insert_units``.
    """
    span_length = 1
    while span_length < synthesis_settings.max_insert_units and noise_generator.random() < 0.5:
        span_length += 1
    return span_length


def stray_letter(unit_texts: list[str], noise_generator: random.Random) -> list[str]:
    """One letter of the segment's own alphabet, standing alone.

    Args:
        unit_texts: Units of the segment.
        noise_generator: Random generator.

    Returns:
        One unit (a stray mark when the segment has no letters).
    """
    segment_letters = [word_char for unit_text in unit_texts for word_char in unit_text if word_char.isalpha()]
    return [noise_generator.choice(segment_letters)] if segment_letters else [noise_generator.choice(STRAY_MARKS)]


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

    Usage:
        Called by ``noise_synthesis`` for a segment picked for an insertion.

    Args:
        unit_texts: Units of the segment before the insertion.
        insert_position: Where the span goes.
        donor_units: Units of another passage of the same book (may be empty).
        synthesis_settings: Span length limit.
        noise_generator: Random generator.

    Returns:
        Non-empty list of units.
    """
    span_kind = noise_generator.choices(SPAN_KINDS, SPAN_WEIGHTS)[0]
    span_length = run_length(synthesis_settings, noise_generator)
    if span_kind == "repeat" and insert_position >= span_length:
        return unit_texts[insert_position - span_length:insert_position]
    if span_kind == "donor" and len(donor_units) > span_length:
        donor_start = noise_generator.randrange(len(donor_units) - span_length)
        return donor_units[donor_start:donor_start + span_length]
    if span_kind == "letter":
        return stray_letter(unit_texts, noise_generator)
    if span_kind == "stray":
        return [noise_generator.choice(STRAY_MARKS)]
    if span_kind == "footnote":
        return noise_generator.choice(FOOTNOTE_MARKERS).split()
    return reader_note(noise_generator)
