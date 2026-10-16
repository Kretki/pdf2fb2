"""Debug overlay: reader marks in red, protected printed lines in green, flagged patches as a grid."""

import cv2
import numpy

from readermarks.inference.spotter_inference import SpotterOutput
from readermarks.postprocess.mask_fusion import FusedMarks

MARK_COLOR = (230, 0, 0)
PROTECTED_COLOR = (0, 180, 0)
GRID_COLOR = (40, 90, 255)


def draw_flagged_grid(overlay_image: numpy.ndarray, spotter_output: SpotterOutput, patch_threshold: float) -> None:
    """Outline every patch whose mark probability reaches the threshold.

    Args:
        overlay_image: (H, W, 3) uint8 image, mutated.
        spotter_output: Stage 1 output.
        patch_threshold: Patch probability threshold.

    Returns:
        None.
    """
    native_patch = spotter_output.grid_geometry.native_patch_size
    for row_index, column_index in zip(*numpy.nonzero(spotter_output.mark_probability >= patch_threshold), strict=True):
        top_left = (int(column_index * native_patch), int(row_index * native_patch))
        bottom_right = (int((column_index + 1) * native_patch) - 1, int((row_index + 1) * native_patch) - 1)
        cv2.rectangle(overlay_image, top_left, bottom_right, GRID_COLOR, 1)


def build_debug_overlay(
    rgb_image: numpy.ndarray, spotter_output: SpotterOutput, fused_marks: FusedMarks | None, patch_threshold: float
) -> numpy.ndarray:
    """Compose the debug overlay of one page.

    Usage:
        Written as ``debug.png`` when the cleaner runs with ``--debug``.

    Args:
        rgb_image: (H, W, 3) uint8 page.
        spotter_output: Stage 1 output.
        fused_marks: Fusion result, or None when Stage 2 did not run.
        patch_threshold: Patch probability threshold.

    Returns:
        (H, W, 3) uint8 overlay.
    """
    overlay_image = rgb_image.copy()
    if fused_marks is not None:
        overlay_image[fused_marks.mark_mask] = MARK_COLOR
        overlay_image[fused_marks.protected_mask] = PROTECTED_COLOR
    draw_flagged_grid(overlay_image, spotter_output, patch_threshold)
    return overlay_image
