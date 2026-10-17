"""Glyph measurement with OpenCV: letter heights inside segments and letter-like ink outside them.

The page is binarised with Otsu's threshold and split into 8-connected ink components. A component counts
as a glyph when it is big enough not to be dust, not much wider than tall (rules, underlines and merged
words are skipped), dense enough not to be a frame or a stroke outline, and not taller than half the page
(scan shadows along the binding).
"""

import statistics
from pathlib import Path

import cv2
import numpy as np

from pdf2fb2.segments.font_statistics import PageGlyphs

MINIMUM_GLYPH_AREA = 8  # pixels
MINIMUM_GLYPH_HEIGHT = 3  # pixels
MAXIMUM_GLYPH_ASPECT = 3.0  # width / height
MINIMUM_GLYPH_FILL = 0.15  # ink pixels / box area
MAXIMUM_GLYPH_PAGE_SHARE = 0.5  # glyph height / page height
MINIMUM_SEGMENT_GLYPHS = 10
COMPONENT_CONNECTIVITY = 8
GlyphBox = tuple[int, int, int, int]


def read_ink_mask(image_path: Path) -> np.ndarray:
    """Binarise a page image into an ink mask.

    Args:
        image_path: Rendered page image.

    Returns:
        uint8 array (height x width): 255 for ink, 0 for paper.

    Raises:
        ValueError: If the image cannot be read.
    """
    gray_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray_image is None:
        raise ValueError(f"cannot read page image {image_path}")
    _, ink_mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return ink_mask


def is_glyph_like(component_stats: np.ndarray, page_height: int) -> bool:
    """Tell whether an ink component has the shape of a letter.

    Args:
        component_stats: One row of ``cv2.connectedComponentsWithStats`` statistics (x, y, width, height, area).
        page_height: Height of the analysed image in pixels.

    Returns:
        True for components passing the area, height, aspect, fill and page-share limits.
    """
    glyph_width = int(component_stats[cv2.CC_STAT_WIDTH])
    glyph_height = int(component_stats[cv2.CC_STAT_HEIGHT])
    ink_area = int(component_stats[cv2.CC_STAT_AREA])
    return (ink_area >= MINIMUM_GLYPH_AREA and MINIMUM_GLYPH_HEIGHT <= glyph_height
            <= MAXIMUM_GLYPH_PAGE_SHARE * page_height and glyph_width <= MAXIMUM_GLYPH_ASPECT * glyph_height
            and ink_area >= MINIMUM_GLYPH_FILL * glyph_width * glyph_height)


def glyph_boxes(ink_mask: np.ndarray, page_height: int) -> list[GlyphBox]:
    """Letter-like ink components of a mask.

    Args:
        ink_mask: uint8 ink mask (255 = ink).
        page_height: Height of the whole page in pixels, for the page-share limit.

    Returns:
        [x1, y1, x2, y2] boxes of the glyphs, in mask pixels.
    """
    _, _, component_table, _ = cv2.connectedComponentsWithStats(ink_mask, connectivity=COMPONENT_CONNECTIVITY)
    return [(int(component_stats[cv2.CC_STAT_LEFT]), int(component_stats[cv2.CC_STAT_TOP]),
             int(component_stats[cv2.CC_STAT_LEFT] + component_stats[cv2.CC_STAT_WIDTH]),
             int(component_stats[cv2.CC_STAT_TOP] + component_stats[cv2.CC_STAT_HEIGHT]))
            for component_stats in component_table[1:] if is_glyph_like(component_stats, page_height)]


def pixel_window(bounding_box: list[float], mask_shape: tuple[int, ...]) -> tuple[slice, slice]:
    """Array slices of a box clamped to an image.

    Args:
        bounding_box: [x1, y1, x2, y2] in pixels.
        mask_shape: Shape of the image array (height, width).

    Returns:
        (row slice, column slice); empty slices for a box outside the image.
    """
    mask_height, mask_width = mask_shape[:2]
    left_edge = min(max(int(round(bounding_box[0])), 0), mask_width)
    top_edge = min(max(int(round(bounding_box[1])), 0), mask_height)
    right_edge = min(max(int(round(bounding_box[2])), left_edge), mask_width)
    bottom_edge = min(max(int(round(bounding_box[3])), top_edge), mask_height)
    return slice(top_edge, bottom_edge), slice(left_edge, right_edge)


def segment_glyph_height(ink_mask: np.ndarray, bounding_box: list[float]) -> float | None:
    """Median glyph height inside one segment box.

    Args:
        ink_mask: uint8 ink mask of the whole page.
        bounding_box: [x1, y1, x2, y2] of the segment in page pixels.

    Returns:
        Median height in pixels, or None when fewer than ``MINIMUM_SEGMENT_GLYPHS`` glyphs are found.
    """
    row_window, column_window = pixel_window(bounding_box, ink_mask.shape)
    segment_mask = np.ascontiguousarray(ink_mask[row_window, column_window])
    if segment_mask.size == 0:
        return None
    glyph_heights = [glyph_box[3] - glyph_box[1] for glyph_box in glyph_boxes(segment_mask, ink_mask.shape[0])]
    if len(glyph_heights) < MINIMUM_SEGMENT_GLYPHS:
        return None
    return float(statistics.median(glyph_heights))


def touches_border(glyph_box: GlyphBox, mask_shape: tuple[int, ...]) -> bool:
    """Tell whether a component touches the page edge (scan borders, cut-off shadows).

    Args:
        glyph_box: [x1, y1, x2, y2] in page pixels.
        mask_shape: Shape of the page mask (height, width).

    Returns:
        True if the box reaches any page edge.
    """
    return glyph_box[0] <= 0 or glyph_box[1] <= 0 or glyph_box[2] >= mask_shape[1] or glyph_box[3] >= mask_shape[0]


def uncovered_glyph_boxes(ink_mask: np.ndarray, covered_boxes: list[list[float]]) -> list[GlyphBox]:
    """Glyphs that lie outside every given box and away from the page edges.

    Args:
        ink_mask: uint8 ink mask of the whole page (not modified).
        covered_boxes: [x1, y1, x2, y2] boxes of the page segments, in page pixels.

    Returns:
        [x1, y1, x2, y2] page-pixel boxes of the uncovered glyphs.
    """
    uncovered_mask = ink_mask.copy()
    for covered_box in covered_boxes:
        row_window, column_window = pixel_window(covered_box, uncovered_mask.shape)
        uncovered_mask[row_window, column_window] = 0
    return [glyph_box for glyph_box in glyph_boxes(uncovered_mask, uncovered_mask.shape[0])
            if not touches_border(glyph_box, uncovered_mask.shape)]


class OpenCvGlyphMeter:
    """Measures glyph heights of the segments of a page and finds letter-like ink outside them.

    Usage:
        Injected into ``PictureRecovery``; call ``measure_page`` for every page record.
    """

    def measure_page(self, page_record: dict) -> PageGlyphs:
        """Measure one segmented page.

        Args:
            page_record: Page record with ``image`` and ``segments``.

        Returns:
            Segment glyph heights (aligned with the segments) and the uncovered glyph boxes.

        Raises:
            ValueError: If the page image cannot be read.
        """
        ink_mask = read_ink_mask(Path(page_record["image"]))
        segment_boxes = [page_segment["bbox"] for page_segment in page_record["segments"]]
        return PageGlyphs(tuple(segment_glyph_height(ink_mask, segment_box) for segment_box in segment_boxes),
                          tuple(uncovered_glyph_boxes(ink_mask, segment_boxes)))
