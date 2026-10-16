"""Margin symbols (tick, asterisk, cross, arrow, dash) as hand-drawn polylines."""

from collections.abc import Callable

import numpy

from readermarks.synthesis.stroke_geometry import Polyline, straightish_line


def tick_paths(
    center_x: float, center_y: float, half_size: float, random_source: numpy.random.Generator
) -> list[Polyline]:
    """A check-mark tick.

    Args:
        center_x: Symbol center x.
        center_y: Symbol center y.
        half_size: Half the symbol size in pixels.
        random_source: Random generator (unused; keeps the builder signature uniform).

    Returns:
        One polyline.
    """
    tick_points = [
        [center_x - half_size, center_y],
        [center_x - half_size * 0.3, center_y + half_size],
        [center_x + half_size, center_y - half_size * 1.2],
    ]
    return [numpy.array(tick_points, numpy.float32)]


def spoke_paths(spoke_count: int) -> Callable:
    """Build a symbol builder of ``spoke_count`` crossing strokes (3 = asterisk, 2 = cross).

    Args:
        spoke_count: Number of strokes through the center.

    Returns:
        Builder ``(center_x, center_y, half_size, random_source) -> polylines``.
    """

    def crossing_strokes(
        center_x: float, center_y: float, half_size: float, random_source: numpy.random.Generator
    ) -> list[Polyline]:
        """Crossing strokes through a center; see ``spoke_paths``."""
        spoke_angles = random_source.uniform(0, numpy.pi) + numpy.arange(spoke_count) * numpy.pi / spoke_count
        return [
            straightish_line(
                (center_x - half_size * numpy.cos(line_angle), center_y - half_size * numpy.sin(line_angle)),
                (center_x + half_size * numpy.cos(line_angle), center_y + half_size * numpy.sin(line_angle)),
                0.3,
                random_source,
            )
            for line_angle in spoke_angles
        ]

    return crossing_strokes


def arrow_paths(
    center_x: float, center_y: float, half_size: float, random_source: numpy.random.Generator
) -> list[Polyline]:
    """A right-pointing arrow (shaft and head).

    Args:
        center_x: Arrow tip region center x.
        center_y: Arrow center y.
        half_size: Half the symbol size in pixels.
        random_source: Random generator.

    Returns:
        Two polylines.
    """
    arrow_shaft = straightish_line(
        (center_x - half_size * 3.0, center_y), (center_x + half_size, center_y), 0.6, random_source
    )
    head_points = [
        [center_x, center_y - half_size * 0.7],
        [center_x + half_size, center_y],
        [center_x, center_y + half_size * 0.7],
    ]
    return [arrow_shaft, numpy.array(head_points, numpy.float32)]


def dash_paths(
    center_x: float, center_y: float, half_size: float, random_source: numpy.random.Generator
) -> list[Polyline]:
    """A short horizontal dash.

    Args:
        center_x: Dash center x.
        center_y: Dash center y.
        half_size: Half the dash length in pixels.
        random_source: Random generator.

    Returns:
        One polyline.
    """
    return [
        straightish_line((center_x - half_size, center_y), (center_x + half_size, center_y + 1), 0.4, random_source)
    ]


SYMBOL_BUILDERS = (tick_paths, spoke_paths(3), spoke_paths(2), arrow_paths, dash_paths)


def symbol_paths(
    center_x: float, center_y: float, symbol_size: float, random_source: numpy.random.Generator
) -> list[Polyline]:
    """A tick, asterisk, cross, arrow or short dash used as a margin mark.

    Usage:
        Called by the margin-mark planner; new symbols are added to ``SYMBOL_BUILDERS``.

    Args:
        center_x: Symbol center x.
        center_y: Symbol center y.
        symbol_size: Approximate size in pixels.
        random_source: Random generator.

    Returns:
        One to four polylines.
    """
    symbol_builder = SYMBOL_BUILDERS[int(random_source.integers(len(SYMBOL_BUILDERS)))]
    return symbol_builder(center_x, center_y, symbol_size / 2, random_source)
