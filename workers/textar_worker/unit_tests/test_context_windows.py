"""Unit tests of the deterministic TexTAR context windows."""

import numpy

from textar_worker.context_windows import WINDOW_SIZE, build_context_windows, window_positions


def grid_boxes(row_count: int, column_count: int) -> list[tuple[int, int, int, int]]:
    """Word boxes on a regular text grid.

    Args:
        row_count: Number of lines.
        column_count: Words per line.

    Returns:
        Boxes in reading order.
    """
    return [(column_index * 60, row_index * 30, column_index * 60 + 50, row_index * 30 + 20)
            for row_index in range(row_count) for column_index in range(column_count)]


def test_small_page_is_padded_cyclically() -> None:
    """A page with fewer words than a window repeats its words."""
    context_windows = build_context_windows(grid_boxes(2, 5))
    assert len(context_windows) == 1
    assert len(context_windows[0]) == WINDOW_SIZE
    assert set(context_windows[0].tolist()) == set(range(10))


def test_large_page_is_fully_covered() -> None:
    """Every word of a long page lands in at least one full window."""
    word_boxes = grid_boxes(40, 10)
    context_windows = build_context_windows(word_boxes)
    covered_words = set(numpy.concatenate(context_windows).tolist())
    assert covered_words == set(range(len(word_boxes)))
    assert all(len(window_indexes) == WINDOW_SIZE for window_indexes in context_windows)


def test_empty_page_has_no_windows() -> None:
    """No words, no windows."""
    assert build_context_windows([]) == []


def test_positions_are_normalised() -> None:
    """Window coordinates span 0..1."""
    word_boxes = grid_boxes(3, 3)
    window_coordinates = window_positions(word_boxes, numpy.arange(len(word_boxes)))
    assert window_coordinates.min() >= 0.0
    assert window_coordinates.max() <= 1.0
