"""Procedural printed pictures as hard negatives: engraving-style hatching and colour cover art.

Book illustrations are full of short strokes that resemble pen marks; the spotter must learn
that they are printed content. Pictures are labelled ``printed_ink`` (dark pixels), never as marks.
"""

import cv2
import numpy

from readermarks.synthesis.page_analysis import PageAnalysis, PixelBox


def smooth_field(
    field_height: int, field_width: int, blur_sigma: float, random_source: numpy.random.Generator
) -> numpy.ndarray:
    """Low-frequency random field normalised to 0..1.

    Args:
        field_height: Height in pixels.
        field_width: Width in pixels.
        blur_sigma: Gaussian blur sigma (feature size) in pixels.
        random_source: Random generator.

    Returns:
        (h, w) float32 field.
    """
    random_field = cv2.GaussianBlur(
        random_source.normal(0, 1, (field_height, field_width)).astype(numpy.float32), (0, 0), blur_sigma
    )
    field_range = random_field.max() - random_field.min()
    return (random_field - random_field.min()) / (field_range if field_range > 0 else 1.0)


def hatching_coverage(field_height: int, field_width: int, random_source: numpy.random.Generator) -> numpy.ndarray:
    """Engraving coverage: tone-dependent hatching layers, contour strokes and stipple.

    Args:
        field_height: Height in pixels.
        field_width: Width in pixels.
        random_source: Random generator.

    Returns:
        (h, w) float32 ink coverage 0..1.
    """
    tone_field = smooth_field(field_height, field_width, random_source.uniform(15, 45), random_source)
    row_grid, column_grid = numpy.mgrid[0:field_height, 0:field_width].astype(numpy.float32)
    coverage = numpy.zeros((field_height, field_width), numpy.float32)
    warp_field = smooth_field(field_height, field_width, 25, random_source) * random_source.uniform(4, 14)
    for layer_index in range(int(random_source.integers(1, 4))):
        line_angle = random_source.uniform(0, numpy.pi)
        stripe_period = random_source.uniform(3.0, 6.0)
        stripe_pattern = 0.5 + 0.5 * numpy.sin(
            2
            * numpy.pi
            * (column_grid * numpy.cos(line_angle) + row_grid * numpy.sin(line_angle) + warp_field)
            / stripe_period
        )
        layer_tone = numpy.clip(tone_field * 1.4 - 0.25 * layer_index, 0, 1)
        coverage = numpy.maximum(coverage, (stripe_pattern < layer_tone * 0.7).astype(numpy.float32))
    stipple_dots = (random_source.random((field_height, field_width)) < 0.04 * tone_field).astype(numpy.float32)
    coverage = numpy.maximum(coverage, stipple_dots)
    contour_layer = numpy.zeros((field_height, field_width), numpy.uint8)
    for _ in range(int(random_source.integers(5, 25))):
        start_point = numpy.array([random_source.uniform(0, field_width), random_source.uniform(0, field_height)])
        walk_steps = numpy.cumsum(random_source.normal(0, 4, (int(random_source.integers(15, 80)), 2)), axis=0)
        contour_points = (start_point + walk_steps).astype(numpy.int32).reshape(-1, 1, 2)
        cv2.polylines(contour_layer, [contour_points], False, 255, int(random_source.integers(1, 3)), cv2.LINE_AA)
    return numpy.maximum(coverage, contour_layer.astype(numpy.float32) / 255.0)


def cover_art(field_height: int, field_width: int, random_source: numpy.random.Generator) -> numpy.ndarray:
    """Colourful painting-like image (smooth multi-scale colour noise with texture).

    Args:
        field_height: Height in pixels.
        field_width: Width in pixels.
        random_source: Random generator.

    Returns:
        (h, w, 3) float32 image 0..1.
    """
    colour_layers = []
    for _ in range(3):
        coarse_field = smooth_field(field_height, field_width, random_source.uniform(20, 80), random_source)
        fine_field = smooth_field(field_height, field_width, random_source.uniform(1.5, 4), random_source)
        colour_layers.append(0.75 * coarse_field + 0.25 * fine_field)
    art_image = numpy.stack(colour_layers, axis=2)
    darkness = smooth_field(field_height, field_width, 40, random_source)[..., None]
    return numpy.clip(art_image * (0.35 + 0.9 * darkness), 0, 1).astype(numpy.float32)


def picture_box(analysis: PageAnalysis, page_shape: tuple[int, int], random_source: numpy.random.Generator) -> PixelBox:
    """Where the picture goes: the whole page (cover), or a block inside the text area.

    Args:
        analysis: Page analysis.
        page_shape: (height, width).
        random_source: Random generator.

    Returns:
        Picture box.
    """
    page_height, page_width = page_shape
    if random_source.random() < 0.25:
        return (0, 0, page_width, page_height)
    text_left, text_top, text_right, text_bottom = analysis.text_box
    box_width = int((text_right - text_left) * random_source.uniform(0.3, 1.0))
    box_height = int((text_bottom - text_top) * random_source.uniform(0.15, 0.9))
    left_edge = int(random_source.uniform(text_left, max(text_left + 1, text_right - box_width)))
    top_edge = int(random_source.uniform(text_top, max(text_top + 1, text_bottom - box_height)))
    return (left_edge, top_edge, left_edge + max(16, box_width), top_edge + max(16, box_height))


def add_printed_picture(
    page_image: numpy.ndarray, ink_mask: numpy.ndarray, analysis: PageAnalysis, random_source: numpy.random.Generator
) -> PixelBox:
    """Paint an engraving or a colour picture over part of the page and label its ink.

    Usage:
        Called by the sample generator for a share of the pages, before marks are added.

    Args:
        page_image: (H, W, 3) float32 page 0..1, mutated.
        ink_mask: (H, W) uint8 printed-ink mask, mutated (picture area relabelled).
        analysis: Page analysis.
        random_source: Random generator.

    Returns:
        The picture box.
    """
    left_edge, top_edge, right_edge, bottom_edge = picture_box(analysis, ink_mask.shape, random_source)
    box_height, box_width = bottom_edge - top_edge, right_edge - left_edge
    paper_rgb = numpy.array(analysis.paper_color, numpy.float32) / 255.0
    if random_source.random() < 0.7:
        coverage = hatching_coverage(box_height, box_width, random_source)
        ink_colour = numpy.array(analysis.ink_color, numpy.float32) / 255.0
        picture_rgb = paper_rgb * (1 - coverage[..., None]) + ink_colour * coverage[..., None]
        picture_ink = coverage > 0.5
    else:
        picture_rgb = cover_art(box_height, box_width, random_source)
        picture_ink = picture_rgb.mean(axis=2) < 0.35
    page_image[top_edge:bottom_edge, left_edge:right_edge] = picture_rgb
    ink_mask[top_edge:bottom_edge, left_edge:right_edge] = numpy.where(picture_ink, 255, 0).astype(numpy.uint8)
    return (left_edge, top_edge, right_edge, bottom_edge)
