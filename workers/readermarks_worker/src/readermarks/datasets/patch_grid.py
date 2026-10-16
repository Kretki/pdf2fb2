"""Geometry of the Stage 1 patch grid and conversions between pixel masks and patch labels."""

from dataclasses import dataclass

import cv2
import numpy

PADDING_VALUE = 255


@dataclass(frozen=True)
class GridGeometry:
    """How a page maps onto the spotter input and its patch grid.

    Attributes:
        page_height: Native page height in pixels.
        page_width: Native page width in pixels.
        scale_factor: Resize factor native -> spotter input (long side = configured length).
        resized_height: Height after resizing, before padding.
        resized_width: Width after resizing, before padding.
        patch_size: ViT patch size in spotter-input pixels.
        grid_rows: Patch rows (padded height / patch size).
        grid_columns: Patch columns (padded width / patch size).
    """

    page_height: int
    page_width: int
    scale_factor: float
    resized_height: int
    resized_width: int
    patch_size: int
    grid_rows: int
    grid_columns: int

    @property
    def native_patch_size(self) -> float:
        """Return the side of one patch in native page pixels.

        Returns:
            ``patch_size / scale_factor``.
        """
        return self.patch_size / self.scale_factor


def compute_geometry(page_height: int, page_width: int, long_side: int, patch_size: int) -> GridGeometry:
    """Compute the spotter input size and patch grid for a page.

    Args:
        page_height: Native page height in pixels (> 0).
        page_width: Native page width in pixels (> 0).
        long_side: Target long side of the spotter input in pixels.
        patch_size: ViT patch size in pixels.

    Returns:
        Grid geometry; the padded size is a multiple of ``patch_size``.
    """
    scale_factor = long_side / max(page_height, page_width)
    resized_height = max(patch_size, round(page_height * scale_factor))
    resized_width = max(patch_size, round(page_width * scale_factor))
    return GridGeometry(
        page_height,
        page_width,
        scale_factor,
        resized_height,
        resized_width,
        patch_size,
        -(-resized_height // patch_size),
        -(-resized_width // patch_size),
    )


def prepare_spotter_image(rgb_image: numpy.ndarray, grid_geometry: GridGeometry) -> numpy.ndarray:
    """Resize and pad a page for the spotter backbone (no binarisation).

    Args:
        rgb_image: Page (H, W, 3) uint8.
        grid_geometry: Output of ``compute_geometry`` for this page.

    Returns:
        Array (rows*patch, columns*patch, 3) uint8, padded with white at the right/bottom.
    """
    resized_image = cv2.resize(
        rgb_image, (grid_geometry.resized_width, grid_geometry.resized_height), interpolation=cv2.INTER_AREA
    )
    padded_height = grid_geometry.grid_rows * grid_geometry.patch_size
    padded_width = grid_geometry.grid_columns * grid_geometry.patch_size
    return cv2.copyMakeBorder(
        resized_image,
        0,
        padded_height - grid_geometry.resized_height,
        0,
        padded_width - grid_geometry.resized_width,
        cv2.BORDER_CONSTANT,
        value=(PADDING_VALUE,) * 3,
    )


def mask_to_patch_fraction(binary_mask: numpy.ndarray, grid_geometry: GridGeometry) -> numpy.ndarray:
    """Compute the covered fraction of every patch for one native-resolution mask.

    Args:
        binary_mask: Mask (H, W), non-zero = class present, native resolution.
        grid_geometry: Geometry of the page.

    Returns:
        Array (grid_rows, grid_columns) float32 with values 0..1.
    """
    resized_mask = cv2.resize(
        (binary_mask > 0).astype(numpy.float32),
        (grid_geometry.resized_width, grid_geometry.resized_height),
        interpolation=cv2.INTER_AREA,
    )
    padded_mask = numpy.zeros(
        (grid_geometry.grid_rows * grid_geometry.patch_size, grid_geometry.grid_columns * grid_geometry.patch_size),
        dtype=numpy.float32,
    )
    padded_mask[: grid_geometry.resized_height, : grid_geometry.resized_width] = resized_mask
    patch_size = grid_geometry.patch_size
    return padded_mask.reshape(grid_geometry.grid_rows, patch_size, grid_geometry.grid_columns, patch_size).mean(
        axis=(1, 3)
    )


def masks_to_patch_labels(
    class_masks: numpy.ndarray, grid_geometry: GridGeometry, label_fraction: float
) -> numpy.ndarray:
    """Turn per-class pixel masks into binary patch labels.

    Usage:
        Used for synthetic pages and for converted real labels.

    Args:
        class_masks: Array (C, H, W) uint8 at native resolution.
        grid_geometry: Geometry of the page.
        label_fraction: A patch is positive when at least this fraction of it is covered, 0..1.

    Returns:
        Array (C, grid_rows, grid_columns) uint8 with values 0/1.
    """
    patch_labels = [mask_to_patch_fraction(class_mask, grid_geometry) >= label_fraction for class_mask in class_masks]
    return numpy.stack(patch_labels).astype(numpy.uint8)


def patch_box_to_pixels(
    row_start: int, final_row: int, column_start: int, column_end: int, grid_geometry: GridGeometry
) -> tuple[int, int, int, int]:
    """Convert a patch-index box (end exclusive) to a native pixel box clipped to the page.

    Args:
        row_start: First patch row.
        final_row: Patch row after the last one.
        column_start: First patch column.
        column_end: Patch column after the last one.
        grid_geometry: Geometry of the page.

    Returns:
        ``(x_start, y_start, x_end, y_end)`` in native pixels (end exclusive).
    """
    native_patch = grid_geometry.native_patch_size
    return (
        max(0, int(column_start * native_patch)),
        max(0, int(row_start * native_patch)),
        min(grid_geometry.page_width, int(numpy.ceil(column_end * native_patch))),
        min(grid_geometry.page_height, int(numpy.ceil(final_row * native_patch))),
    )
