"""Preview sheets: synthetic samples side by side with their colour-coded masks (realism check)."""

from pathlib import Path

import cv2
import numpy

from readermarks.datasets.image_files import read_rgb_image, write_rgb_image
from readermarks.datasets.mask_archive import read_mask_archive
from readermarks.settings.class_catalog import SPOTTER_CLASSES

CLASS_COLORS = {
    "printed_ink": (200, 200, 200),
    "printed_rule": (0, 170, 0),
    "hand_underline": (230, 0, 0),
    "hand_encirclement": (255, 130, 0),
    "margin_mark": (200, 0, 200),
    "hand_text_inline": (0, 60, 255),
    "hand_text_margin": (0, 190, 230),
    "highlighter": (230, 200, 0),
    "strike_through": (120, 0, 160),
}
TILE_HEIGHT = 640
SAMPLES_PER_SHEET = 10
SHEET_COLUMNS = 5


def colorize_masks(class_masks: numpy.ndarray) -> numpy.ndarray:
    """Paint every class in its colour on white; protected classes first so marks stay on top.

    Args:
        class_masks: (C, H, W) masks in ``SPOTTER_CLASSES`` order.

    Returns:
        (H, W, 3) uint8 RGB visualisation.
    """
    mask_view = numpy.full(class_masks.shape[1:] + (3,), 255, numpy.uint8)
    paint_order = sorted(
        range(len(SPOTTER_CLASSES)),
        key=lambda class_index: SPOTTER_CLASSES[class_index] not in ("printed_ink", "printed_rule"),
    )
    for class_index in paint_order:
        mask_view[class_masks[class_index] > 0] = CLASS_COLORS[SPOTTER_CLASSES[class_index]]
    return mask_view


def sample_tile(split_directory: Path, sample_name: str) -> numpy.ndarray:
    """Image and mask visualisation of one sample, side by side, scaled to ``TILE_HEIGHT``.

    Args:
        split_directory: Directory of the split.
        sample_name: ``sample_NNNNNN``.

    Returns:
        (TILE_HEIGHT, w, 3) uint8 RGB tile.
    """
    page_image = read_rgb_image(split_directory / f"{sample_name}.png")
    mask_view = colorize_masks(read_mask_archive(split_directory / f"{sample_name}_masks.npz"))
    pair_image = numpy.hstack([page_image, mask_view])
    tile_width = round(pair_image.shape[1] * TILE_HEIGHT / pair_image.shape[0])
    preview_tile = cv2.resize(pair_image, (tile_width, TILE_HEIGHT), interpolation=cv2.INTER_AREA)
    cv2.putText(preview_tile, sample_name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    return preview_tile


def assemble_sheet(preview_tiles: list[numpy.ndarray]) -> numpy.ndarray:
    """Arrange tiles in rows of ``SHEET_COLUMNS``.

    Args:
        preview_tiles: Tiles of equal height.

    Returns:
        (rows*TILE_HEIGHT, W, 3) uint8 sheet padded with white.
    """
    tile_width = max(preview_tile.shape[1] for preview_tile in preview_tiles)
    padded_tiles = [
        cv2.copyMakeBorder(
            preview_tile,
            0,
            0,
            0,
            tile_width - preview_tile.shape[1],
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )
        for preview_tile in preview_tiles
    ]
    while len(padded_tiles) % SHEET_COLUMNS:
        padded_tiles.append(numpy.full_like(padded_tiles[0], 255))
    sheet_rows = [
        numpy.hstack(padded_tiles[row_start : row_start + SHEET_COLUMNS])
        for row_start in range(0, len(padded_tiles), SHEET_COLUMNS)
    ]
    return numpy.vstack(sheet_rows)


def write_preview_sheets(split_directory: Path, sample_names: list[str], output_directory: Path) -> list[Path]:
    """Write preview sheets of ``SAMPLES_PER_SHEET`` samples each.

    Usage:
        Called by the ``preview`` command (50 samples by default).

    Args:
        split_directory: Directory of the generated split.
        sample_names: Samples to show.
        output_directory: Where ``preview_NN.png`` sheets are written.

    Returns:
        Paths of the written sheets.
    """
    sheet_paths = []
    for sheet_index, sheet_start in enumerate(range(0, len(sample_names), SAMPLES_PER_SHEET)):
        sheet_tiles = [
            sample_tile(split_directory, sample_name)
            for sample_name in sample_names[sheet_start : sheet_start + SAMPLES_PER_SHEET]
        ]
        sheet_path = output_directory / f"preview_{sheet_index:02d}.png"
        write_rgb_image(sheet_path, assemble_sheet(sheet_tiles))
        sheet_paths.append(sheet_path)
    return sheet_paths
