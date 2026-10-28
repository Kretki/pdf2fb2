"""Synthetic training examples: reference segments corrupted the way OCR corrupts real pages.

The kinds and proportions follow statistics of the training split's real output (never its text):

* misread units (label ``misread``): a confusion substitution, a dropped diacritic or a letter edit
  (``unit_misreadings``), two units merged by a lost space, or a word split in two whose second piece is
  ``inserted`` (``span_corruptions``);
* inserted spans (label ``inserted``): mostly single footnote markers, stray letters and marks, plus notes and
  repeated or remembered runs (``inserted_spans``);
* reference units that look like OCR errors of the reference itself are ``ignore`` (``reference_screening``).

Every other unit keeps the label ``correct``.
"""

import random

from token_verifier.datasets.example_files import IGNORE_LABEL, TokenExample
from token_verifier.datasets.inserted_spans import inserted_span
from token_verifier.datasets.reference_screening import is_suspicious_unit
from token_verifier.datasets.span_corruptions import CorruptionResult, merged_units, split_unit
from token_verifier.datasets.unit_misreadings import misread_unit
from token_verifier.verifier_settings import SynthesisSettings

MERGE_SHARE = 0.15
SPLIT_SHARE = 0.10


def corrupted_units(reference_units: list[str], unit_index: int,
                    noise_generator: random.Random) -> CorruptionResult:
    """One misreading at a position: merged, split or single-unit.

    Args:
        reference_units: Units of the segment.
        unit_index: Position of the corrupted unit (non-empty, not suspicious).
        noise_generator: Random generator.

    Returns:
        Labelled replacement units and the number of reference units consumed.
    """
    unit_text = reference_units[unit_index]
    next_unit = reference_units[unit_index + 1] if unit_index + 1 < len(reference_units) else ""
    kind_draw = noise_generator.random()
    boundary_result = None
    if kind_draw < MERGE_SHARE and not is_suspicious_unit(next_unit):
        boundary_result = merged_units(unit_text, next_unit)
    elif kind_draw < MERGE_SHARE + SPLIT_SHARE:
        boundary_result = split_unit(unit_text, noise_generator)
    return boundary_result or CorruptionResult(((misread_unit(unit_text, noise_generator), "misread"),), 1)


def labelled_reference(reference_units: list[str], synthesis_settings: SynthesisSettings,
                       noise_generator: random.Random) -> list[tuple[str, str]]:
    """Units of a segment after misreadings, with their labels.

    Args:
        reference_units: Units of the segment (taken as correct unless suspicious).
        synthesis_settings: Misread rate.
        noise_generator: Random generator.

    Returns:
        ``(unit text, label)`` pairs.
    """
    labelled_pairs: list[tuple[str, str]] = []
    unit_index = 0
    while unit_index < len(reference_units):
        unit_text = reference_units[unit_index]
        if is_suspicious_unit(unit_text):
            labelled_pairs.append((unit_text, IGNORE_LABEL))
            unit_index += 1
            continue
        if unit_text and noise_generator.random() < synthesis_settings.misread_rate:
            corruption_result = corrupted_units(reference_units, unit_index, noise_generator)
            labelled_pairs.extend(corruption_result.labelled_units)
            unit_index += corruption_result.consumed_units
            continue
        labelled_pairs.append((unit_text, "correct"))
        unit_index += 1
    return labelled_pairs


def corrupt_chunk(book_key: str, language_name: str, reference_units: list[str], donor_units: list[str],
                  synthesis_settings: SynthesisSettings, noise_generator: random.Random) -> TokenExample:
    """Build one synthetic example from a reference segment.

    Usage:
        Called by ``synthesis_service`` for every segment of every corrupted copy of a chunk.

    Args:
        book_key: Source book.
        language_name: Corpus language.
        reference_units: Units of the segment.
        donor_units: Units of another chunk of the same book.
        synthesis_settings: Rates and limits.
        noise_generator: Random generator.

    Returns:
        The labelled example.
    """
    labelled_pairs = labelled_reference(reference_units, synthesis_settings, noise_generator)
    if labelled_pairs and noise_generator.random() < synthesis_settings.insert_rate:
        insert_position = noise_generator.randrange(len(labelled_pairs) + 1)
        span_units = inserted_span([unit_text for unit_text, _ in labelled_pairs], insert_position, donor_units,
                                   synthesis_settings, noise_generator)
        labelled_pairs[insert_position:insert_position] = [(span_unit, "inserted") for span_unit in span_units]
    return TokenExample(book_key, language_name, tuple(unit_text for unit_text, _ in labelled_pairs),
                        tuple(unit_label for _, unit_label in labelled_pairs))
