"""Label-side analysis of a clean page: printed-ink mask, printed rules, text lines and words.

The adaptive threshold runs on a copy of the page and is used only to build labels and to
place marks; it is never fed to a model (STAGE_CLEAN.md §3, §7.1).
"""

from dataclasses import dataclass

import cv2
import numpy

PixelBox = tuple[int, int, int, int]  # (x_start, y_start, x_end, y_end), end exclusive

INK_BLOCK_SIZE = 31
INK_OFFSET = 18
MINIMUM_SPECK_PIXELS = 3
RULE_LENGTH_FACTOR = 6.0
RULE_MAXIMUM_THICKNESS = 6


@dataclass(frozen=True)
class PageAnalysis:
    """Everything the generator needs to know about a clean page.

    Attributes:
        ink_mask: (H, W) uint8 0/255 printed-ink mask (labels only).
        rule_mask: (H, W) uint8 0/255 printed rules (long thin straight lines).
        word_boxes: Word boxes in reading order of their lines.
        line_boxes: Text-line boxes, top to bottom.
        text_box: Bounding box of all text lines (the text block); whole page when empty.
        x_height: Estimated typical glyph height in pixels.
        ink_color: Median RGB color of printed ink.
        paper_color: Median RGB color of paper.
    """

    ink_mask: numpy.ndarray
    rule_mask: numpy.ndarray
    word_boxes: list[PixelBox]
    line_boxes: list[PixelBox]
    text_box: PixelBox
    x_height: float
    ink_color: tuple[int, int, int]
    paper_color: tuple[int, int, int]


def compute_ink_mask(gray_image: numpy.ndarray) -> numpy.ndarray:
    """Adaptive-threshold printed ink on a grayscale copy, without small specks.

    Args:
        gray_image: (H, W) uint8 grayscale page.

    Returns:
        (H, W) uint8 mask, 255 = ink.
    """
    ink_mask = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, INK_BLOCK_SIZE, INK_OFFSET
    )
    component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)
    keep_component = component_stats[:, cv2.CC_STAT_AREA] >= MINIMUM_SPECK_PIXELS
    keep_component[0] = False
    return numpy.where(keep_component[component_labels], 255, 0).astype(numpy.uint8)


def estimate_x_height(ink_mask: numpy.ndarray) -> float:
    """Estimate the typical glyph height from connected components.

    Args:
        ink_mask: Output of ``compute_ink_mask``.

    Returns:
        Median component height of glyph-sized components, defaults to 10 px.
    """
    _, _, component_stats, _ = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)
    glyph_heights = component_stats[1:, cv2.CC_STAT_HEIGHT]
    glyph_heights = glyph_heights[(glyph_heights >= 4) & (glyph_heights <= 60)]
    return float(numpy.median(glyph_heights)) if glyph_heights.size else 10.0


def detect_rules(ink_mask: numpy.ndarray, x_height: float) -> numpy.ndarray:
    """Find long, thin, straight horizontal and vertical printed lines.

    Args:
        ink_mask: Output of ``compute_ink_mask``.
        x_height: Output of ``estimate_x_height``.

    Returns:
        (H, W) uint8 mask, 255 = printed rule.
    """
    rule_length = max(25, int(RULE_LENGTH_FACTOR * x_height))
    horizontal_lines = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, numpy.ones((1, rule_length), numpy.uint8))
    vertical_lines = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, numpy.ones((rule_length, 1), numpy.uint8))
    candidate_lines = cv2.bitwise_or(horizontal_lines, vertical_lines)
    component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(candidate_lines)
    thickness = numpy.minimum(component_stats[:, cv2.CC_STAT_WIDTH], component_stats[:, cv2.CC_STAT_HEIGHT])
    keep_component = thickness <= RULE_MAXIMUM_THICKNESS
    keep_component[0] = False
    rule_mask = numpy.where(keep_component[component_labels], 255, 0).astype(numpy.uint8)
    return cv2.bitwise_and(cv2.dilate(rule_mask, numpy.ones((3, 3), numpy.uint8)), ink_mask)


def merged_boxes(ink_mask: numpy.ndarray, kernel_width: int, kernel_height: int, x_height: float) -> list[PixelBox]:
    """Merge ink into blobs with a rectangular dilation and return plausible text boxes.

    Args:
        ink_mask: Printed-ink mask.
        kernel_width: Horizontal dilation in pixels (word or line scale).
        kernel_height: Vertical dilation in pixels.
        x_height: Typical glyph height, used to reject noise and pictures.

    Returns:
        Boxes sorted top-to-bottom, then left-to-right.
    """
    merged_mask = cv2.dilate(ink_mask, numpy.ones((kernel_height, kernel_width), numpy.uint8))
    _, _, component_stats, _ = cv2.connectedComponentsWithStats(merged_mask, connectivity=8)
    text_boxes = []
    for corner_x, corner_y, box_width, box_height, _ in component_stats[1:]:
        if 0.6 * x_height <= box_height <= 4.5 * x_height and box_width >= 0.8 * x_height:
            text_boxes.append((int(corner_x), int(corner_y), int(corner_x + box_width), int(corner_y + box_height)))
    return sorted(text_boxes, key=lambda text_box: (text_box[1] // max(1, int(x_height)), text_box[0]))


def union_box(box_list: list[PixelBox], fallback_box: PixelBox) -> PixelBox:
    """Return the bounding box of several boxes.

    Args:
        box_list: Boxes to unite.
        fallback_box: Returned when ``boxes`` is empty.

    Returns:
        Union box.
    """
    if not box_list:
        return fallback_box
    box_array = numpy.array(box_list)
    return (
        int(box_array[:, 0].min()),
        int(box_array[:, 1].min()),
        int(box_array[:, 2].max()),
        int(box_array[:, 3].max()),
    )


def median_colors(
    rgb_image: numpy.ndarray, ink_mask: numpy.ndarray
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Median RGB of ink pixels and of paper pixels.

    Args:
        rgb_image: Page (H, W, 3) uint8.
        ink_mask: Printed-ink mask.

    Returns:
        ``(ink_color, paper_color)``; ink defaults to dark gray on an empty page.
    """
    ink_pixels = rgb_image[ink_mask > 0]
    paper_pixels = rgb_image[cv2.dilate(ink_mask, numpy.ones((5, 5), numpy.uint8)) == 0]
    ink_color = (
        tuple(int(channel_value) for channel_value in numpy.median(ink_pixels, axis=0))
        if len(ink_pixels)
        else (40, 40, 40)
    )
    paper_color = (
        tuple(int(channel_value) for channel_value in numpy.median(paper_pixels, axis=0))
        if len(paper_pixels)
        else (235,) * 3
    )
    return ink_color, paper_color


def analyse_page(rgb_image: numpy.ndarray) -> PageAnalysis:
    """Analyse a clean page for label building and mark placement.

    Usage:
        Called once per base page by the synthetic generator.

    Args:
        rgb_image: Page (H, W, 3) uint8 RGB.

    Returns:
        Page analysis with ink and rule masks, line/word boxes and colors.
    """
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    ink_mask = compute_ink_mask(gray_image)
    x_height = estimate_x_height(ink_mask)
    word_boxes = merged_boxes(ink_mask, max(3, int(0.45 * x_height)), 3, x_height)
    line_boxes = merged_boxes(ink_mask, max(9, int(2.2 * x_height)), 3, x_height)
    page_box = (0, 0, rgb_image.shape[1], rgb_image.shape[0])
    ink_color, paper_color = median_colors(rgb_image, ink_mask)
    return PageAnalysis(
        ink_mask,
        detect_rules(ink_mask, x_height),
        word_boxes,
        line_boxes,
        union_box(line_boxes, page_box),
        x_height,
        ink_color,
        paper_color,
    )
