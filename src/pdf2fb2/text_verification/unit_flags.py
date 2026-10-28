"""Decisions on verifier probabilities: which units are misread, which are not on the page, how bad a text is."""

from dataclasses import dataclass

CORRECT_POSITION = 0
MISREAD_POSITION = 1
INSERTED_POSITION = 2


@dataclass(frozen=True)
class UnitFlags:
    """Flagged units of one text.

    Attributes:
        misread_units: Units read wrongly (re-read their region).
        inserted_units: Units that are not on the page (hallucinated text, reader notes).
    """

    misread_units: tuple[int, ...]
    inserted_units: tuple[int, ...]


def flag_units(probability_rows: list[list[float]], class_thresholds: dict[str, float]) -> UnitFlags:
    """Flag units whose probability reaches the tuned threshold (``inserted`` wins over ``misread``).

    Args:
        probability_rows: ``[correct, misread, inserted]`` per unit.
        class_thresholds: Thresholds of ``misread`` and ``inserted``.

    Returns:
        The flags.
    """
    inserted_units = tuple(unit_index for unit_index, probability_row in enumerate(probability_rows)
                           if probability_row[INSERTED_POSITION] >= class_thresholds["inserted"])
    misread_units = tuple(unit_index for unit_index, probability_row in enumerate(probability_rows)
                          if probability_row[MISREAD_POSITION] >= class_thresholds["misread"]
                          and unit_index not in inserted_units)
    return UnitFlags(misread_units, inserted_units)


def error_mass(probability_rows: list[list[float]], skipped_units: frozenset[int] = frozenset()) -> float:
    """Mean probability that a unit is wrong.

    Args:
        probability_rows: Probabilities per unit.
        skipped_units: Units left out (e.g. deleted ones).

    Returns:
        Mean of ``1 - p(correct)`` over the other units (0 for none).
    """
    error_values = [1.0 - probability_row[CORRECT_POSITION] for unit_index, probability_row
                    in enumerate(probability_rows) if unit_index not in skipped_units]
    return sum(error_values) / len(error_values) if error_values else 0.0
