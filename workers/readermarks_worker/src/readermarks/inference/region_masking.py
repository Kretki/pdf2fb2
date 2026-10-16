"""Blank excluded regions (printed pictures found by the layout model) with the page's paper colour.

Engravings and photos are full of strokes that look hand-drawn; blanking them before detection
removes that source of false positives. Masking is deterministic and never invents content.
"""

from collections.abc import Sequence

import numpy

REGION_PADDING = 4
PAPER_QUANTILE = 0.5

RegionBox = Sequence[float]


def paper_colour(rgb_image: numpy.ndarray) -> numpy.ndarray:
    """Median colour of the brighter half of the page (paper, not ink or pictures).

    Args:
        rgb_image: (H, W, 3) uint8 page.

    Returns:
        (3,) uint8 RGB colour.
    """
    luminance = rgb_image.mean(axis=2)
    paper_pixels = rgb_image[luminance >= numpy.quantile(luminance, PAPER_QUANTILE)]
    return numpy.median(paper_pixels, axis=0).astype(numpy.uint8)


def blank_regions(rgb_image: numpy.ndarray, region_boxes: Sequence[RegionBox]) -> numpy.ndarray:
    """Copy of the page with every region (plus a small padding) filled with paper colour.

    Usage:
        Called by the cleaning service before a page is inspected; with no regions the input
        array itself is returned.

    Args:
        rgb_image: (H, W, 3) uint8 page.
        region_boxes: ``[x_start, y_start, x_end, y_end]`` pixel boxes.

    Returns:
        (H, W, 3) uint8 page.
    """
    if not region_boxes:
        return rgb_image
    blanked_image = rgb_image.copy()
    fill_colour = paper_colour(rgb_image)
    page_height, page_width = rgb_image.shape[:2]
    for left_edge, top_edge, right_edge, bottom_edge in region_boxes:
        row_slice = slice(max(0, int(top_edge) - REGION_PADDING), min(page_height, int(bottom_edge) + REGION_PADDING))
        column_slice = slice(max(0, int(left_edge) - REGION_PADDING), min(page_width, int(right_edge) + REGION_PADDING))
        blanked_image[row_slice, column_slice] = fill_colour
    return blanked_image
