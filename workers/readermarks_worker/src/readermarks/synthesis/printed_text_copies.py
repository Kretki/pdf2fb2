"""Relocated printed text as hard negatives: headings, captions and letter-spaced capitals.

Real printed lines of the base page are copied (optionally enlarged and letter-spaced) into empty
areas such as margins and gaps, so the models learn that printed glyphs outside the text block,
or with wide letter spacing, are not handwriting. Copies are labelled ``printed_ink``.
"""

import cv2
import numpy

from readermarks.synthesis.page_analysis import PageAnalysis, PixelBox

PAPER_GAP_FRACTION = 0.02


def letter_space(line_crop: numpy.ndarray, ink_crop: numpy.ndarray, extra_spacing: int) -> tuple:
    """Insert paper columns into every ink-free column gap (spreads glyphs apart).

    Args:
        line_crop: (h, w, 3) float32 crop of a printed line.
        ink_crop: (h, w) bool ink of the crop.
        extra_spacing: Paper columns inserted per gap.

    Returns:
        ``(spaced_crop, spaced_ink)``.
    """
    ink_columns = ink_crop.any(axis=0)
    paper_column = numpy.median(line_crop[:, ~ink_columns], axis=1) if (~ink_columns).any() else line_crop[:, 0]
    crop_parts, ink_parts = [], []
    for column_index in range(line_crop.shape[1]):
        crop_parts.append(line_crop[:, column_index : column_index + 1])
        ink_parts.append(ink_crop[:, column_index : column_index + 1])
        gap_starts = column_index > 0 and ink_columns[column_index - 1] and not ink_columns[column_index]
        if gap_starts and extra_spacing > 0:
            crop_parts.append(numpy.repeat(paper_column[:, None], extra_spacing, axis=1))
            ink_parts.append(numpy.zeros((ink_crop.shape[0], extra_spacing), bool))
    return numpy.concatenate(crop_parts, axis=1), numpy.concatenate(ink_parts, axis=1)


def empty_target(
    analysis: PageAnalysis,
    copy_shape: tuple[int, int],
    page_shape: tuple[int, int],
    random_source: numpy.random.Generator,
) -> tuple[int, int] | None:
    """Top-left corner of a place without printed ink for a copy of the given size.

    Args:
        analysis: Page analysis.
        copy_shape: (height, width) of the copy.
        page_shape: (height, width) of the page.
        random_source: Random generator.

    Returns:
        ``(upper_row, left_column)``, or None when no empty place was found.
    """
    copy_height, copy_width = copy_shape
    page_height, page_width = page_shape
    if copy_height >= page_height or copy_width >= page_width:
        return None
    for _ in range(30):
        upper_row = int(random_source.integers(0, page_height - copy_height))
        left_column = int(random_source.integers(0, page_width - copy_width))
        ink_window = analysis.ink_mask[upper_row : upper_row + copy_height, left_column : left_column + copy_width]
        if (ink_window > 0).mean() <= PAPER_GAP_FRACTION:
            return upper_row, left_column
    return None


def prepared_copy(
    page_image: numpy.ndarray, analysis: PageAnalysis, line_box: PixelBox, random_source: numpy.random.Generator
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Crop a printed line, optionally letter-space and enlarge it.

    Args:
        page_image: (H, W, 3) float32 page.
        analysis: Page analysis.
        line_box: Line to copy.
        random_source: Random generator.

    Returns:
        ``(copy_rgb, copy_ink)``.
    """
    left_edge, top_edge, right_edge, bottom_edge = line_box
    line_crop = page_image[top_edge:bottom_edge, left_edge:right_edge].copy()
    ink_crop = analysis.ink_mask[top_edge:bottom_edge, left_edge:right_edge] > 0
    if random_source.random() < 0.5:
        line_crop, ink_crop = letter_space(line_crop, ink_crop, int(random_source.integers(2, 8)))
    scale_factor = random_source.uniform(0.9, 1.8)
    copy_size = (max(1, int(line_crop.shape[1] * scale_factor)), max(1, int(line_crop.shape[0] * scale_factor)))
    copy_rgb = cv2.resize(line_crop, copy_size, interpolation=cv2.INTER_LINEAR)
    copy_ink = cv2.resize(ink_crop.astype(numpy.uint8), copy_size, interpolation=cv2.INTER_NEAREST) > 0
    return copy_rgb, copy_ink


def add_printed_text_copies(
    page_image: numpy.ndarray, ink_mask: numpy.ndarray, analysis: PageAnalysis, random_source: numpy.random.Generator
) -> None:
    """Paste 1..3 relocated printed lines into empty areas (darken-only blend) and label their ink.

    Usage:
        Called by the sample generator for a share of the pages, before marks are added.

    Args:
        page_image: (H, W, 3) float32 page 0..1, mutated.
        ink_mask: (H, W) uint8 printed-ink mask, mutated.
        analysis: Page analysis.
        random_source: Random generator.

    Returns:
        None.
    """
    if not analysis.line_boxes:
        return
    for _ in range(int(random_source.integers(1, 4))):
        line_box = analysis.line_boxes[int(random_source.integers(len(analysis.line_boxes)))]
        copy_rgb, copy_ink = prepared_copy(page_image, analysis, line_box, random_source)
        target_corner = empty_target(analysis, copy_ink.shape, ink_mask.shape, random_source)
        if target_corner is None:
            continue
        upper_row, left_column = target_corner
        rows_slice = slice(upper_row, upper_row + copy_ink.shape[0])
        columns_slice = slice(left_column, left_column + copy_ink.shape[1])
        page_image[rows_slice, columns_slice] = numpy.minimum(page_image[rows_slice, columns_slice], copy_rgb)
        ink_mask[rows_slice, columns_slice][copy_ink] = 255
