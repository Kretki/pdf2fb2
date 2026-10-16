"""Rasterise stroke polylines and handwriting into local alpha maps (coverage 0..1).

Strokes are drawn 4x supersampled with a pressure profile (varying width and opacity) and
tapered endpoints, then area-downsampled, so edges are anti-aliased like a real scan.
"""

from dataclasses import dataclass

import cv2
import numpy

from readermarks.synthesis.stroke_geometry import Polyline, densify_polyline, smooth_jitter

SUPERSAMPLING = 4
CANVAS_MARGIN = 12


@dataclass(frozen=True)
class LocalAlpha:
    """Alpha (coverage) of one mark in a local window of the page.

    Attributes:
        alpha_map: (h, w) float32 values 0..1.
        x_offset: Page x of the window's left column.
        y_offset: Page y of the window's top row.
    """

    alpha_map: numpy.ndarray
    x_offset: int
    y_offset: int


def pressure_profile(point_count: int, random_source: numpy.random.Generator) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Width multipliers (with tapered ends) and opacities along a stroke.

    Args:
        point_count: Number of polyline points (>= 2).
        random_source: Random generator.

    Returns:
        ``(width_factors, opacities)`` arrays of shape (point_count,), values 0..1.
    """
    position = numpy.linspace(0.0, 1.0, point_count)
    taper_length = random_source.uniform(0.04, 0.2)
    taper_profile = numpy.clip(numpy.minimum(position, 1.0 - position) / taper_length, 0.0, 1.0) ** 0.6
    width_factors = (0.35 + 0.65 * taper_profile) * (1.0 + smooth_jitter(point_count, 0.25, random_source))
    opacities = numpy.clip(0.8 + smooth_jitter(point_count, 0.2, random_source), 0.45, 1.0) * (
        0.55 + 0.45 * taper_profile
    )
    return numpy.clip(width_factors, 0.2, 1.5), opacities


def draw_polyline(
    draw_canvas: numpy.ndarray, scaled_points: numpy.ndarray, base_width: float, random_source: numpy.random.Generator
) -> None:
    """Draw one supersampled polyline with its pressure profile onto a uint8 canvas.

    Args:
        draw_canvas: Supersampled (H, W) uint8 canvas, mutated in place (anti-aliased overdraw).
        scaled_points: Polyline already in canvas coordinates.
        base_width: Stroke width in page pixels.
        random_source: Random generator.

    Returns:
        None.
    """
    width_factors, opacities = pressure_profile(len(scaled_points), random_source)
    integer_points = numpy.round(scaled_points).astype(int)
    for point_index in range(len(scaled_points) - 1):
        segment_thickness = max(1, round(base_width * width_factors[point_index] * SUPERSAMPLING))
        cv2.line(
            draw_canvas,
            tuple(integer_points[point_index]),
            tuple(integer_points[point_index + 1]),
            int(255 * opacities[point_index]),
            segment_thickness,
            cv2.LINE_AA,
        )


def render_polylines(polylines: list[Polyline], base_width: float, random_source: numpy.random.Generator) -> LocalAlpha:
    """Rasterise polylines into one local alpha map.

    Usage:
        Called for every procedural mark (underline, circle, bracket, ...).

    Args:
        polylines: Paths in page pixels.
        base_width: Stroke width in page pixels (about 0.8..4).
        random_source: Random generator.

    Returns:
        Local alpha map covering all polylines plus a margin.
    """
    polylines = [densify_polyline(polyline) for polyline in polylines]
    all_points = numpy.concatenate(polylines)
    x_offset = int(numpy.floor(all_points[:, 0].min())) - CANVAS_MARGIN
    y_offset = int(numpy.floor(all_points[:, 1].min())) - CANVAS_MARGIN
    window_width = int(numpy.ceil(all_points[:, 0].max())) + CANVAS_MARGIN - x_offset
    window_height = int(numpy.ceil(all_points[:, 1].max())) + CANVAS_MARGIN - y_offset
    draw_canvas = numpy.zeros((window_height * SUPERSAMPLING, window_width * SUPERSAMPLING), numpy.uint8)
    for polyline in polylines:
        scaled_points = (polyline - numpy.array([x_offset, y_offset], numpy.float32)) * SUPERSAMPLING
        if len(scaled_points) >= 2:
            draw_polyline(draw_canvas, scaled_points, base_width, random_source)
    alpha_map = cv2.resize(draw_canvas, (window_width, window_height), interpolation=cv2.INTER_AREA)
    return LocalAlpha(alpha_map.astype(numpy.float32) / 255.0, x_offset, y_offset)


def elastic_warp(
    alpha_map: numpy.ndarray, warp_strength: float, random_source: numpy.random.Generator
) -> numpy.ndarray:
    """Smooth random displacement field applied to a handwriting alpha map.

    Args:
        alpha_map: (h, w) float32 alpha.
        warp_strength: Maximum displacement in pixels.
        random_source: Random generator.

    Returns:
        Warped alpha map with the same shape.
    """
    map_height, map_width = alpha_map.shape
    displacement = random_source.normal(0, 1, (2, map_height, map_width)).astype(numpy.float32)
    displacement = numpy.stack([cv2.GaussianBlur(noise_field, (0, 0), 6) for noise_field in displacement])
    displacement *= warp_strength / (numpy.abs(displacement).max() + 1e-6)
    row_grid, column_grid = numpy.mgrid[0:map_height, 0:map_width].astype(numpy.float32)
    return cv2.remap(
        alpha_map,
        column_grid + displacement[0],
        row_grid + displacement[1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


HERSHEY_FONTS = (
    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_PLAIN,
    cv2.FONT_HERSHEY_SIMPLEX | cv2.FONT_ITALIC,
    cv2.FONT_HERSHEY_COMPLEX_SMALL | cv2.FONT_ITALIC,
)
BLOCK_LETTER_FONTS = (
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_PLAIN,
    cv2.FONT_HERSHEY_SIMPLEX | cv2.FONT_ITALIC,
    cv2.FONT_HERSHEY_DUPLEX,
)


def jittered_text_canvas(
    note_text: str, font_face: int, font_scale: float, thickness: int, random_source: numpy.random.Generator
) -> numpy.ndarray:
    """Write text character by character with jittered size, baseline and spacing.

    Args:
        note_text: Text to write (ASCII).
        font_face: Hershey font id.
        font_scale: Base OpenCV font scale (supersampled).
        thickness: Pen thickness in supersampled pixels.
        random_source: Random generator.

    Returns:
        Supersampled (H, W) uint8 canvas with the text in white on black.
    """
    (_, text_height), _ = cv2.getTextSize("Hg", font_face, font_scale, thickness)
    padding_size = int(text_height * 1.2) + thickness * 2
    advance_widths = [
        cv2.getTextSize(character, font_face, font_scale * 1.15, thickness)[0][0] for character in note_text
    ]
    draw_canvas = numpy.zeros(
        (text_height + 2 * padding_size, int(sum(advance_widths) * 1.2) + 2 * padding_size), numpy.uint8
    )
    cursor_x = float(padding_size)
    baseline_drift = smooth_jitter(len(note_text) + 2, 0.12 * text_height, random_source)
    for character_index, character in enumerate(note_text):
        character_scale = font_scale * random_source.uniform(0.88, 1.12)
        baseline_y = (
            padding_size + text_height + baseline_drift[character_index] + random_source.normal(0, 0.04 * text_height)
        )
        cv2.putText(
            draw_canvas,
            character,
            (int(cursor_x), int(baseline_y)),
            font_face,
            character_scale,
            255,
            thickness,
            cv2.LINE_AA,
        )
        cursor_x += advance_widths[character_index] * random_source.uniform(0.82, 1.08)
    return draw_canvas


def transform_note(
    draw_canvas: numpy.ndarray, rotation_degrees: float, random_source: numpy.random.Generator
) -> numpy.ndarray:
    """Shear, rotate, downsample and elastically warp a supersampled note, then crop it tightly.

    Args:
        draw_canvas: Supersampled uint8 note canvas.
        rotation_degrees: Rotation of the note.
        random_source: Random generator.

    Returns:
        (h, w) float32 coverage at page resolution.
    """
    affine_matrix = cv2.getRotationMatrix2D((draw_canvas.shape[1] / 2, draw_canvas.shape[0] / 2), rotation_degrees, 1.0)
    affine_matrix[0, 1] += random_source.uniform(-0.35, 0.25)
    bound_size = int(numpy.hypot(*draw_canvas.shape)) + 2
    affine_matrix[:, 2] += (bound_size - numpy.array([draw_canvas.shape[1], draw_canvas.shape[0]])) / 2
    warped_canvas = cv2.warpAffine(draw_canvas, affine_matrix, (bound_size, bound_size), flags=cv2.INTER_LINEAR)
    small_map = (
        cv2.resize(
            warped_canvas, (bound_size // SUPERSAMPLING, bound_size // SUPERSAMPLING), interpolation=cv2.INTER_AREA
        ).astype(numpy.float32)
        / 255.0
    )
    small_map = elastic_warp(small_map, random_source.uniform(0.5, 2.0), random_source)
    ink_rows, ink_columns = numpy.nonzero(small_map > 0.05)
    if ink_rows.size == 0:
        return numpy.zeros((1, 1), numpy.float32)
    return small_map[ink_rows.min() : ink_rows.max() + 1, ink_columns.min() : ink_columns.max() + 1]


def render_handwriting(
    note_text: str,
    letter_height: float,
    stroke_width: float,
    rotation_degrees: float,
    random_source: numpy.random.Generator,
) -> numpy.ndarray:
    """Render a handwritten-looking note (jittered Hershey stroke font, sheared, rotated, warped).

    Usage:
        Called by the inline and margin handwriting planners.

    Args:
        note_text: Text to write (ASCII).
        letter_height: Target height of capital letters in page pixels.
        stroke_width: Pen width in page pixels.
        rotation_degrees: Rotation of the whole note (e.g. 90 for a vertical margin note).
        random_source: Random generator.

    Returns:
        (h, w) float32 alpha map, tightly cropped around the ink.
    """
    font_choices = BLOCK_LETTER_FONTS if note_text.isupper() else HERSHEY_FONTS
    font_face = int(random_source.choice(font_choices))
    font_scale = letter_height * SUPERSAMPLING / 22.0
    thickness = max(1, round(stroke_width * SUPERSAMPLING))
    draw_canvas = jittered_text_canvas(note_text, font_face, font_scale, thickness, random_source)
    return transform_note(draw_canvas, rotation_degrees, random_source)
