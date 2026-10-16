"""Fixed-length context windows of neighbouring words (deterministic version of TexTAR's data selection).

The official pipeline picks a random anchor word and takes its 125 nearest words under a weighted
Chebyshev distance with random weights (vertical offsets weighted about 4x more than horizontal
ones). Here the anchor is the first not yet covered word and the weights are fixed to the means
of that sampling, so results are reproducible. Pages with fewer than 125 words repeat their words
cyclically to fill the window (the model needs exactly 125 tokens).
"""

from collections.abc import Sequence

import numpy

WINDOW_SIZE = 125
VERTICAL_WEIGHT = 24.0
HORIZONTAL_WEIGHT = 6.0

PixelWordBox = Sequence[int]


def box_centres(word_boxes: Sequence[PixelWordBox]) -> numpy.ndarray:
    """Centres of the word boxes.

    Args:
        word_boxes: ``(x0, y0, x1, y1)`` boxes.

    Returns:
        ``(N, 2)`` float array of ``(x, y)`` centres.
    """
    box_array = numpy.asarray(word_boxes, dtype=numpy.float64).reshape(-1, 4)
    return numpy.stack([(box_array[:, 0] + box_array[:, 2]) / 2, (box_array[:, 1] + box_array[:, 3]) / 2], axis=1)


def nearest_words(centre_array: numpy.ndarray, anchor_index: int, window_size: int) -> numpy.ndarray:
    """Indexes of the ``window_size`` words closest to the anchor (anchor first).

    Args:
        centre_array: ``(N, 2)`` word centres, N >= window_size.
        anchor_index: Index of the anchor word.
        window_size: Number of words to return.

    Returns:
        ``(window_size,)`` int array.
    """
    centre_offsets = numpy.abs(centre_array - centre_array[anchor_index])
    word_distances = numpy.maximum(HORIZONTAL_WEIGHT * centre_offsets[:, 0], VERTICAL_WEIGHT * centre_offsets[:, 1])
    return numpy.argsort(word_distances, kind="stable")[:window_size]


def build_context_windows(word_boxes: Sequence[PixelWordBox], window_size: int = WINDOW_SIZE) -> list[numpy.ndarray]:
    """Cover every word of a page by at least one context window.

    Usage:
        Call once per page; predictions of words that fall into several windows are averaged.

    Args:
        word_boxes: Word boxes of the page in reading order.
        window_size: Tokens per window (125 for the released checkpoint).

    Returns:
        List of ``(window_size,)`` int arrays of word indexes; empty for a page without words.
    """
    word_count = len(word_boxes)
    if word_count == 0:
        return []
    if word_count <= window_size:
        return [numpy.arange(window_size) % word_count]
    centre_array = box_centres(word_boxes)
    covered_words = numpy.zeros(word_count, dtype=bool)
    context_windows = []
    while not covered_words.all():
        window_indexes = nearest_words(centre_array, int(numpy.flatnonzero(~covered_words)[0]), window_size)
        covered_words[window_indexes] = True
        context_windows.append(window_indexes)
    return context_windows


def window_positions(word_boxes: Sequence[PixelWordBox], window_indexes: numpy.ndarray) -> numpy.ndarray:
    """Word centres normalised to 0..1 by the extent of the window (as in the official data loader).

    Args:
        word_boxes: Word boxes of the page.
        window_indexes: Word indexes of one window.

    Returns:
        ``(window_size, 2)`` float32 array of ``(x, y)``.
    """
    box_array = numpy.asarray(word_boxes, dtype=numpy.float64).reshape(-1, 4)[window_indexes]
    window_start = numpy.array([box_array[:, [0, 2]].min(), box_array[:, [1, 3]].min()])
    window_extent = numpy.array([box_array[:, [0, 2]].max(), box_array[:, [1, 3]].max()]) - window_start
    window_extent = numpy.maximum(window_extent, 1.0)
    return ((box_centres(box_array) - window_start) / window_extent).astype(numpy.float32)
