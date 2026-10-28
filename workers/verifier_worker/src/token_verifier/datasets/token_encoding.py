"""Units -> sub-word windows for the encoder, and sub-word predictions -> one prediction per unit.

Long texts are cut into windows of ``window_units`` units. For training the windows do not overlap; for
prediction they overlap by half, and every unit takes its prediction from the window where it is farthest from
an edge (most context on both sides). Only the first sub-word of a unit carries its label.
"""

from dataclasses import dataclass

from token_verifier.datasets.example_files import IGNORE_INDEX, TokenExample, label_index


@dataclass(frozen=True)
class EncodedWindow:
    """One encoder input.

    Attributes:
        input_identifiers: Sub-word ids (special tokens included).
        label_identifiers: Class index per sub-word (``IGNORE_INDEX`` except on first sub-words).
        unit_positions: Sub-word position of the first piece of every unit of the window (None if truncated).
        first_unit: Index of the window's first unit in the whole text.
    """

    input_identifiers: tuple[int, ...]
    label_identifiers: tuple[int, ...]
    unit_positions: tuple[int | None, ...]
    first_unit: int


def window_ranges(unit_count: int, window_units: int, overlapping: bool) -> list[tuple[int, int]]:
    """Start and end unit of every window.

    Args:
        unit_count: Units of the text.
        window_units: Units per window.
        overlapping: Half-overlapping windows (prediction) or disjoint ones (training).

    Returns:
        ``(start, end)`` pairs covering every unit.
    """
    step_units = max(1, window_units // 2) if overlapping else window_units
    window_starts = list(range(0, max(unit_count - window_units, 0) + 1, step_units))
    if window_starts[-1] + window_units < unit_count:
        window_starts.append(unit_count - window_units)
    return [(window_start, min(unit_count, window_start + window_units)) for window_start in window_starts]


def encode_window(text_tokenizer, unit_texts: list[str], unit_labels: list[str] | None, first_unit: int,
                  max_tokens: int) -> EncodedWindow:
    """Encode the units of one window.

    Args:
        text_tokenizer: Fast Hugging Face tokenizer (``word_ids`` support).
        unit_texts: Units of the window.
        unit_labels: Their labels, or None for prediction.
        first_unit: Index of the first unit in the whole text.
        max_tokens: Sub-word limit (longer windows are truncated).

    Returns:
        The encoded window.
    """
    encoding = text_tokenizer(unit_texts, is_split_into_words=True, truncation=True, max_length=max_tokens)
    unit_positions: list[int | None] = [None] * len(unit_texts)
    label_identifiers = []
    for token_position, unit_index in enumerate(encoding.word_ids()):
        is_first_piece = unit_index is not None and unit_positions[unit_index] is None
        if is_first_piece:
            unit_positions[unit_index] = token_position
        label_identifiers.append(label_index(unit_labels[unit_index]) if is_first_piece and unit_labels
                                 else IGNORE_INDEX)
    return EncodedWindow(tuple(encoding["input_ids"]), tuple(label_identifiers), tuple(unit_positions), first_unit)


def training_windows(text_tokenizer, token_example: TokenExample, window_units: int,
                     max_tokens: int) -> list[EncodedWindow]:
    """Disjoint labelled windows of one example.

    Args:
        text_tokenizer: Fast tokenizer.
        token_example: Labelled example.
        window_units: Units per window.
        max_tokens: Sub-word limit.

    Returns:
        Windows (none for an empty example).
    """
    if not token_example.unit_texts:
        return []
    return [encode_window(text_tokenizer, list(token_example.unit_texts[window_start:window_end]),
                          list(token_example.unit_labels[window_start:window_end]), window_start, max_tokens)
            for window_start, window_end in window_ranges(len(token_example.unit_texts), window_units, False)]


def prediction_windows(text_tokenizer, unit_texts: list[str], window_units: int,
                       max_tokens: int) -> list[EncodedWindow]:
    """Overlapping unlabelled windows of one text.

    Args:
        text_tokenizer: Fast tokenizer.
        unit_texts: Units of the text.
        window_units: Units per window.
        max_tokens: Sub-word limit.

    Returns:
        Windows (none for an empty text).
    """
    if not unit_texts:
        return []
    return [encode_window(text_tokenizer, unit_texts[window_start:window_end], None, window_start, max_tokens)
            for window_start, window_end in window_ranges(len(unit_texts), window_units, True)]


def edge_distance(unit_offset: int, window_length: int) -> int:
    """Context of a unit inside its window.

    Args:
        unit_offset: Position of the unit in the window.
        window_length: Units of the window.

    Returns:
        Distance to the nearer window edge.
    """
    return min(unit_offset, window_length - 1 - unit_offset)


def merge_window_predictions(unit_count: int, encoded_windows: list[EncodedWindow],
                             window_probabilities: list[list[list[float]]]) -> list[list[float]]:
    """One probability triple per unit from overlapping windows.

    Args:
        unit_count: Units of the text.
        encoded_windows: Windows of ``prediction_windows``.
        window_probabilities: Per window, per sub-word: class probabilities.

    Returns:
        Triples per unit; a unit truncated out of every window gets ``[1, 0, 0]``.
    """
    best_context = [-1] * unit_count
    unit_probabilities = [[1.0, 0.0, 0.0] for _ in range(unit_count)]
    for encoded_window, token_probabilities in zip(encoded_windows, window_probabilities, strict=True):
        window_length = len(encoded_window.unit_positions)
        for unit_offset, token_position in enumerate(encoded_window.unit_positions):
            unit_index = encoded_window.first_unit + unit_offset
            context_size = edge_distance(unit_offset, window_length)
            if token_position is not None and context_size > best_context[unit_index]:
                best_context[unit_index] = context_size
                unit_probabilities[unit_index] = list(token_probabilities[token_position])
    return unit_probabilities
