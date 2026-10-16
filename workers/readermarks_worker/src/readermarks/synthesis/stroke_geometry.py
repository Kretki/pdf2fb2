"""Procedural hand-drawn stroke paths (polylines in page pixels) placed relative to text boxes.

Every function returns a list of polylines; each polyline is an (N, 2) float32 array of (x, y).
Paths carry low-frequency hand jitter so they are never perfectly straight or closed.
"""

import numpy

from readermarks.synthesis.page_analysis import PixelBox

Polyline = numpy.ndarray


def smooth_jitter(point_count: int, amplitude: float, random_source: numpy.random.Generator) -> numpy.ndarray:
    """Low-frequency random offsets (a smoothed random walk) for hand tremor.

    Args:
        point_count: Number of samples (>= 2).
        amplitude: Approximate maximum offset in pixels.
        random_source: Random generator.

    Returns:
        Array (point_count,) float32.
    """
    random_walk = numpy.cumsum(random_source.normal(0.0, 1.0, point_count))
    kernel_length = min(point_count, max(3, point_count // 6))
    smoothed_walk = numpy.convolve(random_walk, numpy.ones(kernel_length) / kernel_length, mode="same")
    smoothed_walk -= numpy.linspace(smoothed_walk[0], smoothed_walk[-1], point_count)
    peak_value = numpy.abs(smoothed_walk).max() or 1.0
    return (smoothed_walk / peak_value * amplitude).astype(numpy.float32)


def straightish_line(
    start_point: tuple[float, float],
    end_point: tuple[float, float],
    jitter_amplitude: float,
    random_source: numpy.random.Generator,
    wave_amplitude: float = 0.0,
) -> Polyline:
    """A hand-drawn line between two points, optionally wavy.

    Args:
        start_point: (x, y) start.
        end_point: (x, y) end.
        jitter_amplitude: Perpendicular tremor in pixels.
        random_source: Random generator.
        wave_amplitude: Amplitude of a regular wave in pixels (0 = none).

    Returns:
        Polyline (N, 2).
    """
    start_array, end_array = numpy.array(start_point, numpy.float32), numpy.array(end_point, numpy.float32)
    line_length = float(numpy.linalg.norm(end_array - start_array)) or 1.0
    point_count = max(8, int(line_length / 3))
    position = numpy.linspace(0.0, 1.0, point_count, dtype=numpy.float32)
    direction = (end_array - start_array) / line_length
    normal_vector = numpy.array([-direction[1], direction[0]], numpy.float32)
    wave_period = random_source.uniform(10.0, 26.0)
    normal_offsets = smooth_jitter(point_count, jitter_amplitude, random_source)
    normal_offsets += wave_amplitude * numpy.sin(position * line_length / wave_period * 2 * numpy.pi)
    return start_array + position[:, None] * (end_array - start_array) + normal_offsets[:, None] * normal_vector


def underline_paths(span_box: PixelBox, random_source: numpy.random.Generator) -> list[Polyline]:
    """Straight / sloped / wavy / double / partial / overshooting underline below a text span.

    Args:
        span_box: Box of the underlined words.
        random_source: Random generator.

    Returns:
        One or two polylines.
    """
    left_edge, top_edge, right_edge, bottom_edge = span_box
    span_width = right_edge - left_edge
    if random_source.random() < 0.25:
        partial_start = left_edge + random_source.uniform(0, 0.4) * span_width
        right_edge = int(partial_start + random_source.uniform(0.3, 0.6) * span_width)
        left_edge = int(partial_start)
    overshoot_left, overshoot_right = random_source.uniform(-6, 14, 2)
    baseline = bottom_edge + random_source.uniform(0.5, 4.0)
    slope_drop = random_source.uniform(-0.025, 0.025) * span_width
    wave_amplitude = random_source.uniform(1.0, 2.5) if random_source.random() < 0.15 else 0.0
    first_line = straightish_line(
        (left_edge - overshoot_left, baseline),
        (right_edge + overshoot_right, baseline + slope_drop),
        random_source.uniform(0.4, 1.8),
        random_source,
        wave_amplitude,
    )
    if random_source.random() < 0.12:
        double_gap = random_source.uniform(2.5, 5.0)
        return [first_line, first_line + numpy.array([random_source.uniform(-4, 4), double_gap], numpy.float32)]
    return [first_line]


def strike_paths(span_box: PixelBox, random_source: numpy.random.Generator) -> list[Polyline]:
    """A hand-drawn line through the middle of a text span (sometimes a zig-zag scribble).

    Args:
        span_box: Box of the struck words.
        random_source: Random generator.

    Returns:
        One polyline.
    """
    left_edge, top_edge, right_edge, bottom_edge = span_box
    middle_y = (top_edge + bottom_edge) / 2 + random_source.uniform(-2, 2)
    if random_source.random() < 0.2:
        zigzag_x = numpy.linspace(left_edge, right_edge, max(4, (right_edge - left_edge) // 12))
        zigzag_y = middle_y + numpy.where(numpy.arange(len(zigzag_x)) % 2, -1, 1) * (bottom_edge - top_edge) * 0.35
        return [numpy.stack([zigzag_x, zigzag_y], axis=1).astype(numpy.float32)]
    return [
        straightish_line(
            (left_edge - random_source.uniform(0, 6), middle_y),
            (right_edge + random_source.uniform(0, 6), middle_y + random_source.uniform(-3, 3)),
            random_source.uniform(0.4, 1.5),
            random_source,
        )
    ]


def ellipse_path(span_box: PixelBox, random_source: numpy.random.Generator) -> list[Polyline]:
    """An imperfect ellipse (or rounded box) around a span, left unclosed or overlapping.

    Args:
        span_box: Box of the encircled words.
        random_source: Random generator.

    Returns:
        One polyline.
    """
    left_edge, top_edge, right_edge, bottom_edge = span_box
    center_x, center_y = (left_edge + right_edge) / 2, (top_edge + bottom_edge) / 2
    radius_x = (right_edge - left_edge) / 2 + random_source.uniform(3, 12)
    radius_y = (bottom_edge - top_edge) / 2 + random_source.uniform(3, 10)
    start_angle = random_source.uniform(0, 2 * numpy.pi)
    sweep_angle = 2 * numpy.pi + random_source.uniform(-0.6, 0.5)
    point_count = max(40, int((radius_x + radius_y) * 1.5))
    sweep_angles = start_angle + numpy.linspace(0, sweep_angle, point_count)
    radius_scale = 1.0 + smooth_jitter(point_count, random_source.uniform(0.03, 0.10), random_source)
    tilt_angle = random_source.uniform(-0.12, 0.12)
    ellipse_x = radius_x * radius_scale * numpy.cos(sweep_angles)
    ellipse_y = radius_y * radius_scale * numpy.sin(sweep_angles)
    if random_source.random() < 0.25:
        ellipse_x = numpy.clip(ellipse_x * 1.15, -radius_x, radius_x)
        ellipse_y = numpy.clip(ellipse_y * 1.25, -radius_y, radius_y)
    rotated_x = ellipse_x * numpy.cos(tilt_angle) - ellipse_y * numpy.sin(tilt_angle) + center_x
    rotated_y = ellipse_x * numpy.sin(tilt_angle) + ellipse_y * numpy.cos(tilt_angle) + center_y
    return [numpy.stack([rotated_x, rotated_y], axis=1).astype(numpy.float32)]


def bracket_paths(
    margin_x: float, upper_row_y: float, bottom_y: float, facing_right: bool, random_source: numpy.random.Generator
) -> list[Polyline]:
    """A margin bracket, single/double vertical bar or wavy bar beside a range of lines.

    Args:
        margin_x: x position of the vertical stroke.
        upper_row_y: Top of the marked range.
        bottom_y: Bottom of the marked range.
        facing_right: True for a left-margin bracket ``[`` whose hooks point right.
        random_source: Random generator.

    Returns:
        One to three polylines.
    """
    vertical_bar = straightish_line(
        (margin_x, upper_row_y),
        (margin_x + random_source.uniform(-5, 5), bottom_y),
        random_source.uniform(0.8, 2.5),
        random_source,
    )
    style_draw = random_source.random()
    if style_draw < 0.45:
        hook_length = random_source.uniform(5, 16) * (1 if facing_right else -1)
        top_hook = straightish_line(
            (margin_x, upper_row_y),
            (margin_x + hook_length, upper_row_y + random_source.uniform(-2, 3)),
            0.5,
            random_source,
        )
        bottom_hook = straightish_line(
            (vertical_bar[-1][0], bottom_y),
            (vertical_bar[-1][0] + hook_length, bottom_y + random_source.uniform(-3, 2)),
            0.5,
            random_source,
        )
        return [numpy.concatenate([top_hook[::-1], vertical_bar, bottom_hook])]
    if style_draw < 0.65:
        return [vertical_bar, vertical_bar + numpy.array([random_source.uniform(3, 6), 0], numpy.float32)]
    return [vertical_bar]


def densify_polyline(polyline: Polyline, point_spacing: float = 3.0) -> Polyline:
    """Resample a polyline so consecutive points are about ``point_spacing`` pixels apart.

    Usage:
        Called before rasterising, so the pressure profile varies smoothly along long segments.

    Args:
        polyline: (N, 2) points, N >= 1.
        point_spacing: Target spacing in pixels (> 0).

    Returns:
        (M, 2) float32 points including both original endpoints.
    """
    if len(polyline) < 2:
        return polyline.astype(numpy.float32)
    segment_lengths = numpy.linalg.norm(numpy.diff(polyline, axis=0), axis=1)
    cumulative = numpy.concatenate([[0.0], numpy.cumsum(segment_lengths)])
    sample_count = max(2, int(cumulative[-1] / point_spacing) + 1)
    sample_positions = numpy.linspace(0.0, cumulative[-1], sample_count)
    resampled_x = numpy.interp(sample_positions, cumulative, polyline[:, 0])
    resampled_y = numpy.interp(sample_positions, cumulative, polyline[:, 1])
    return numpy.stack([resampled_x, resampled_y], axis=1).astype(numpy.float32)
