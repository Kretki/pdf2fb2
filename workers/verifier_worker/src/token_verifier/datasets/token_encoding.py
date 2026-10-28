"""Texts -> encoder windows with unit labels and evidence, and window predictions -> one prediction per unit.

Long texts are cut into windows of ``window_units`` units. Every window also receives up to ``context_units``
neighbouring units on each side as unlabelled context (the rest of the text, then the page context of the text),
so a short segment is read with its surroundings. For training the windows do not overlap; for prediction they
overlap by half, and every unit takes its prediction from the window where it is farthest from an edge.
Labels, evidence and crop rows are per unit; the network pools all pieces of a unit.
"""

from dataclasses import dataclass

import numpy

from token_verifier.datasets.example_files import IGNORE_INDEX, label_index
from token_verifier.datasets.verify_texts import VerifyText
from token_verifier.features.feature_assembly import UnitEvidence
from token_verifier.networks.backbone_adapters import BackboneAdapter

CORRECT_ROW = (1.0, 0.0, 0.0)


@dataclass(frozen=True)
class EncodedWindow:
    """One encoder input.

    Attributes:
        input_identifiers: Piece ids (context and special pieces included).
        piece_units: Unit index within the window per piece (-1 for others).
        unit_labels: Class index per unit (``IGNORE_INDEX`` for ignore or prediction).
        feature_rows: ``(units, feature_width)`` evidence.
        visual_rows: ``(units, visual_width)`` crop embeddings, or None.
        first_unit: Index of the window's first unit in the whole text.
    """

    input_identifiers: tuple[int, ...]
    piece_units: tuple[int, ...]
    unit_labels: tuple[int, ...]
    feature_rows: numpy.ndarray
    visual_rows: numpy.ndarray | None
    first_unit: int

    @property
    def unit_count(self) -> int:
        """Units of the window.

        Returns:
            Number of units.
        """
        return len(self.unit_labels)

    def encoded_units(self) -> set[int]:
        """Units that kept at least one piece after truncation.

        Returns:
            Unit indexes within the window.
        """
        return {piece_unit for piece_unit in self.piece_units if piece_unit >= 0}


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


def window_context(verify_text: VerifyText, window_start: int, window_end: int,
                   context_units: int) -> tuple[list[str], list[str]]:
    """Unlabelled units around a window: the rest of the text first, then the page context.

    Args:
        verify_text: Whole text.
        window_start: First unit of the window.
        window_end: Unit after the window.
        context_units: Largest context per side (0 disables context).

    Returns:
        ``(before, after)``.
    """
    if context_units <= 0:
        return [], []
    before_units = [*verify_text.context_before, *verify_text.unit_texts[:window_start]][-context_units:]
    after_units = [*verify_text.unit_texts[window_end:], *verify_text.context_after][:context_units]
    return before_units, after_units


def encode_window(backbone_adapter: BackboneAdapter, verify_text: VerifyText, unit_evidence: UnitEvidence,
                  window_bounds: tuple[int, int], max_tokens: int, context_units: int) -> EncodedWindow:
    """Encode the units of one window.

    Args:
        backbone_adapter: Input encoding of the backbone.
        verify_text: Whole text.
        unit_evidence: Evidence of the whole text.
        window_bounds: ``(start, end)`` units.
        max_tokens: Piece limit.
        context_units: Context per side.

    Returns:
        The encoded window.
    """
    window_start, window_end = window_bounds
    before_units, after_units = window_context(verify_text, window_start, window_end, context_units)
    piece_encoding = backbone_adapter.encode_units(list(verify_text.unit_texts[window_start:window_end]),
                                                   before_units, after_units, max_tokens)
    unit_labels = tuple(label_index(unit_label) for unit_label in verify_text.unit_labels[window_start:window_end]) \
        if verify_text.unit_labels else (IGNORE_INDEX,) * (window_end - window_start)
    visual_rows = None if unit_evidence.visual_rows is None else unit_evidence.visual_rows[window_start:window_end]
    return EncodedWindow(piece_encoding.input_identifiers, piece_encoding.piece_units, unit_labels,
                         unit_evidence.feature_rows[window_start:window_end], visual_rows, window_start)


def text_windows(backbone_adapter: BackboneAdapter, verify_text: VerifyText, unit_evidence: UnitEvidence,
                 window_settings: tuple[int, int, int], overlapping: bool) -> list[EncodedWindow]:
    """All windows of one text.

    Usage:
        Training uses ``overlapping=False``, prediction ``overlapping=True``.

    Args:
        backbone_adapter: Input encoding of the backbone.
        verify_text: Text.
        unit_evidence: Its evidence.
        window_settings: ``(window_units, max_tokens, context_units)``.
        overlapping: Half-overlapping windows.

    Returns:
        Windows (none for an empty text).
    """
    window_units, max_tokens, context_units = window_settings
    if not verify_text.unit_texts:
        return []
    return [encode_window(backbone_adapter, verify_text, unit_evidence, window_bounds, max_tokens, context_units)
            for window_bounds in window_ranges(len(verify_text.unit_texts), window_units, overlapping)]


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
        encoded_windows: Windows of ``text_windows(..., overlapping=True)``.
        window_probabilities: Per window, per unit: class probabilities.

    Returns:
        Triples per unit; a unit truncated out of every window gets ``[1, 0, 0]``.
    """
    best_context = [-1] * unit_count
    unit_probabilities = [list(CORRECT_ROW) for _ in range(unit_count)]
    for encoded_window, unit_rows in zip(encoded_windows, window_probabilities, strict=True):
        encoded_units = encoded_window.encoded_units()
        for unit_offset in encoded_units:
            unit_index = encoded_window.first_unit + unit_offset
            context_size = edge_distance(unit_offset, encoded_window.unit_count)
            if context_size > best_context[unit_index]:
                best_context[unit_index] = context_size
                unit_probabilities[unit_index] = list(unit_rows[unit_offset])
    return unit_probabilities
