"""Deterministic mark removal: mark-only pixels become the local paper background, ink is kept.

No generative inpainting is used (STAGE_CLEAN.md §3): the background estimate is the median of
nearby non-ink, non-mark pixels on a tile grid, bilinearly interpolated.
"""

import cv2
import numpy


def tile_medians(
    rgb_image: numpy.ndarray, excluded_mask: numpy.ndarray, tile_pixels: int
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Median color of the usable (paper) pixels of every tile.

    Args:
        rgb_image: (H, W, 3) uint8 page.
        excluded_mask: (H, W) bool, pixels that are not paper (ink, marks).
        tile_pixels: Tile side in pixels.

    Returns:
        ``(medians (rows, columns, 3) float32, valid (rows, columns) bool)``.
    """
    tile_rows = -(-rgb_image.shape[0] // tile_pixels)
    tile_columns = -(-rgb_image.shape[1] // tile_pixels)
    tile_colors = numpy.zeros((tile_rows, tile_columns, 3), numpy.float32)
    valid_tiles = numpy.zeros((tile_rows, tile_columns), bool)
    for row_index in range(tile_rows):
        for column_index in range(tile_columns):
            row_slice = slice(row_index * tile_pixels, (row_index + 1) * tile_pixels)
            column_slice = slice(column_index * tile_pixels, (column_index + 1) * tile_pixels)
            paper_pixels = rgb_image[row_slice, column_slice][~excluded_mask[row_slice, column_slice]]
            if len(paper_pixels) >= 8:
                tile_colors[row_index, column_index] = numpy.median(paper_pixels, axis=0)
                valid_tiles[row_index, column_index] = True
    return tile_colors, valid_tiles


def fill_missing_tiles(tile_colors: numpy.ndarray, valid_tiles: numpy.ndarray) -> numpy.ndarray:
    """Fill tiles without paper pixels from their nearest valid tile.

    Args:
        tile_colors: (rows, columns, 3) tile medians.
        valid_tiles: (rows, columns) bool.

    Returns:
        Filled medians (same shape); a page without any paper tile becomes white.
    """
    if not valid_tiles.any():
        return numpy.full_like(tile_colors, 255.0)
    _, nearest_labels = cv2.distanceTransformWithLabels(
        (~valid_tiles).astype(numpy.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
    )
    valid_rows, valid_columns = numpy.nonzero(valid_tiles)
    label_lookup = numpy.zeros(nearest_labels.max() + 1, numpy.int64)
    label_lookup[nearest_labels[valid_rows, valid_columns]] = numpy.arange(valid_rows.size)
    nearest_index = label_lookup[nearest_labels]
    return tile_colors[valid_rows[nearest_index], valid_columns[nearest_index]]


def estimate_background(rgb_image: numpy.ndarray, excluded_mask: numpy.ndarray, tile_pixels: int) -> numpy.ndarray:
    """Smooth local paper background of the whole page.

    Args:
        rgb_image: (H, W, 3) uint8 page.
        excluded_mask: (H, W) bool, non-paper pixels.
        tile_pixels: Tile side in pixels.

    Returns:
        (H, W, 3) uint8 background estimate.
    """
    tile_colors, valid_tiles = tile_medians(rgb_image, excluded_mask, tile_pixels)
    filled_medians = fill_missing_tiles(tile_colors, valid_tiles)
    tile_rows, tile_columns = valid_tiles.shape
    upsampled = cv2.resize(
        filled_medians, (tile_columns * tile_pixels, tile_rows * tile_pixels), interpolation=cv2.INTER_LINEAR
    )
    return numpy.clip(upsampled[: rgb_image.shape[0], : rgb_image.shape[1]] + 0.5, 0, 255).astype(numpy.uint8)


def removal_mask(mark_mask: numpy.ndarray, ink_mask: numpy.ndarray, dilation_pixels: int) -> numpy.ndarray:
    """Pixels to restore: the dilated mark mask minus printed ink.

    Args:
        mark_mask: (H, W) bool kept reader marks.
        ink_mask: (H, W) bool printed ink (kept even where a mark overlaps it).
        dilation_pixels: Dilation radius that catches anti-aliased stroke edges.

    Returns:
        (H, W) bool mask; always a subset of the dilated mark mask.
    """
    kernel_size = 2 * dilation_pixels + 1
    dilated_marks = cv2.dilate(mark_mask.astype(numpy.uint8), numpy.ones((kernel_size, kernel_size), numpy.uint8)) > 0
    return dilated_marks & ~ink_mask


def remove_marks(
    rgb_image: numpy.ndarray, mark_mask: numpy.ndarray, ink_mask: numpy.ndarray, dilation_pixels: int, tile_pixels: int
) -> numpy.ndarray:
    """Build ``clean.png``: replace mark-only pixels with the local background.

    Usage:
        Called by the page cleaner for pages with marks.

    Args:
        rgb_image: (H, W, 3) uint8 page.
        mark_mask: (H, W) bool kept reader marks.
        ink_mask: (H, W) bool printed ink.
        dilation_pixels: Dilation of the mark mask in pixels.
        tile_pixels: Background tile side in pixels.

    Returns:
        New (H, W, 3) uint8 page; pixels outside the dilated mark mask are untouched.
    """
    restore_mask = removal_mask(mark_mask, ink_mask, dilation_pixels)
    background = estimate_background(rgb_image, restore_mask | ink_mask, tile_pixels)
    cleaned_image = rgb_image.copy()
    cleaned_image[restore_mask] = background[restore_mask]
    return cleaned_image
