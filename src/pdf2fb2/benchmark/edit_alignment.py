"""Edit-distance alignment of output words against reference words, and the error rates built on it.

Every output word is labelled ``correct`` (equal to its reference word), ``misread`` (substituted) or ``inserted``
(no reference word: text the OCR added). Reference words without an output word are ``missed``. The character
error rate is computed on this word alignment: the character distance of every substituted pair plus the
length of inserted and missed words (and their separating space). It equals the exact character distance
whenever errors stay inside words, and runs in O(words²) instead of O(characters²). Substituting two
dissimilar words costs 1.5, so at the edges of a located passage text not in the book stays ``inserted``.
"""

import math
from dataclasses import dataclass

CORRECT_LABEL = "correct"
MISREAD_LABEL = "misread"
INSERTED_LABEL = "inserted"
MISSED_LABEL = "missed"
DISSIMILAR_COST = 1.5


@dataclass(frozen=True)
class AlignedPair:
    """One alignment step.

    Attributes:
        output_index: Index of the output word, or None for a missed reference word.
        reference_index: Index of the reference word, or None for an inserted output word.
        pair_label: ``correct``, ``misread``, ``inserted`` or ``missed``.
    """

    output_index: int | None
    reference_index: int | None
    pair_label: str


def sequence_distance(first_sequence: str, second_sequence: str) -> int:
    """Levenshtein distance of two strings.

    Args:
        first_sequence: First string.
        second_sequence: Second string.

    Returns:
        Number of insertions, deletions and substitutions.
    """
    previous_row = list(range(len(second_sequence) + 1))
    for first_index, first_char in enumerate(first_sequence, start=1):
        current_row = [first_index]
        for second_index, second_char in enumerate(second_sequence, start=1):
            current_row.append(min(previous_row[second_index] + 1, current_row[second_index - 1] + 1,
                                   previous_row[second_index - 1] + (first_char != second_char)))
        previous_row = current_row
    return previous_row[-1]


def substitution_cost(output_word: str, reference_word: str) -> float:
    """Cost of aligning two words as one misread word.

    Args:
        output_word: Output word.
        reference_word: Reference word.

    Returns:
        0 for equal words, 1 for similar ones (same first or last letter, or half of the letters shared),
        ``DISSIMILAR_COST`` otherwise: cheaper than a missed plus an inserted word inside the page, dearer than
        an inserted word at a page edge, where surplus reference words are skipped for free.
    """
    if output_word == reference_word:
        return 0.0
    shared_letters = len(set(output_word) & set(reference_word)) / max(len(set(output_word) | set(reference_word)), 1)
    if output_word[:1] == reference_word[:1] or output_word[-1:] == reference_word[-1:] or shared_letters >= 0.5:
        return 1.0
    return DISSIMILAR_COST


def distance_table(output_words: list[str], reference_words: list[str], free_ends: bool) -> list[list[float]]:
    """Weighted word-level edit table.

    Args:
        output_words: Output words.
        reference_words: Reference words.
        free_ends: Whether leading reference words may be skipped at no cost.

    Returns:
        Table of size (len(output)+1) x (len(reference)+1).
    """
    cost_table = [[0.0] * (len(reference_words) + 1) if free_ends else
                  [float(reference_index) for reference_index in range(len(reference_words) + 1)]]
    for output_index, output_word in enumerate(output_words, start=1):
        previous_row, current_row = cost_table[-1], [float(output_index)]
        for reference_index, reference_word in enumerate(reference_words, start=1):
            current_row.append(min(previous_row[reference_index] + 1, current_row[reference_index - 1] + 1,
                                   previous_row[reference_index - 1] + substitution_cost(output_word, reference_word)))
        cost_table.append(current_row)
    return cost_table


def trace_back(cost_table: list[list[float]], output_words: list[str], reference_words: list[str],
               end_reference: int, free_ends: bool) -> list[AlignedPair]:
    """Alignment steps from the table, ending at a reference position.

    Args:
        cost_table: Table of ``distance_table``.
        output_words: Output words.
        reference_words: Reference words.
        end_reference: Reference position where the alignment ends.
        free_ends: Whether leading reference words are skipped without a step.

    Returns:
        Steps in reading order.
    """
    aligned_pairs: list[AlignedPair] = []
    output_index, reference_index = len(output_words), end_reference
    while output_index or (reference_index and not free_ends):
        current_cost = cost_table[output_index][reference_index]
        if output_index and reference_index:
            pair_cost = substitution_cost(output_words[output_index - 1], reference_words[reference_index - 1])
            if math.isclose(current_cost, cost_table[output_index - 1][reference_index - 1] + pair_cost):
                aligned_pairs.append(AlignedPair(output_index - 1, reference_index - 1,
                                                 CORRECT_LABEL if pair_cost == 0 else MISREAD_LABEL))
                output_index, reference_index = output_index - 1, reference_index - 1
                continue
        if output_index and math.isclose(current_cost, cost_table[output_index - 1][reference_index] + 1):
            aligned_pairs.append(AlignedPair(output_index - 1, None, INSERTED_LABEL))
            output_index -= 1
            continue
        aligned_pairs.append(AlignedPair(None, reference_index - 1, MISSED_LABEL))
        reference_index -= 1
    return aligned_pairs[::-1]


def align_words(output_words: list[str], reference_words: list[str], free_ends: bool = False) -> list[AlignedPair]:
    """Minimal word alignment of output against reference.

    Usage:
        Used for page scores and for the verifier's training labels; pass ``free_ends`` when the reference is
        a located passage with margins.

    Args:
        output_words: Normalised output words.
        reference_words: Normalised reference words.
        free_ends: Skip reference words before and after the aligned part at no cost (they get no step).

    Returns:
        Alignment steps in reading order.
    """
    cost_table = distance_table(output_words, reference_words, free_ends)
    last_row = cost_table[-1]
    end_reference = min(range(len(last_row)), key=lambda reference_index: (last_row[reference_index],
                                                                            -reference_index)) \
        if free_ends else len(reference_words)
    return trace_back(cost_table, output_words, reference_words, end_reference, free_ends)


def character_errors(aligned_pairs: list[AlignedPair], output_words: list[str], reference_words: list[str]) -> int:
    """Character edits implied by a word alignment.

    Args:
        aligned_pairs: Alignment of the two word lists.
        output_words: Output words.
        reference_words: Reference words.

    Returns:
        Character distance (substitutions inside words, whole inserted/missed words plus one space each).
    """
    error_count = 0
    for aligned_pair in aligned_pairs:
        if aligned_pair.pair_label == MISREAD_LABEL:
            error_count += sequence_distance(output_words[aligned_pair.output_index],
                                             reference_words[aligned_pair.reference_index])
        elif aligned_pair.pair_label == INSERTED_LABEL:
            error_count += len(output_words[aligned_pair.output_index]) + 1
        elif aligned_pair.pair_label == MISSED_LABEL:
            error_count += len(reference_words[aligned_pair.reference_index]) + 1
    return error_count
