"""Stage 2 inference: tile the flagged regions with native-resolution crops and predict pixel masks."""

import numpy
import torch
from torch import nn

from readermarks.datasets.segmenter_dataset import PADDING_VALUE, normalise_crop, padded_crop
from readermarks.synthesis.page_analysis import PixelBox

TILE_OVERLAP = 96
TILE_BATCH = 4


def tile_origins(region_start: int, region_end: int, crop_size: int) -> list[int]:
    """Start coordinates of crops covering ``[region_start, region_end)`` along one axis.

    Args:
        region_start: First pixel of the region.
        region_end: Pixel after the region.
        crop_size: Crop side.

    Returns:
        Sorted crop starts (one crop when the region fits).
    """
    region_length = region_end - region_start
    if region_length <= crop_size:
        return [region_start + (region_length - crop_size) // 2]
    tile_stride = crop_size - TILE_OVERLAP
    tile_count = -(-(region_length - crop_size) // tile_stride) + 1
    return [int(position) for position in numpy.linspace(region_start, region_end - crop_size, tile_count)]


def region_tiles(region_boxes: list[PixelBox], crop_size: int) -> list[tuple[int, int, PixelBox]]:
    """All crops needed for the regions, each tagged with the region it serves.

    Args:
        region_boxes: Flagged regions.
        crop_size: Crop side.

    Returns:
        List of ``(row_start, column_start, region_box)``.
    """
    return [
        (row_start, column_start, region_box)
        for region_box in region_boxes
        for row_start in tile_origins(region_box[1], region_box[3], crop_size)
        for column_start in tile_origins(region_box[0], region_box[2], crop_size)
    ]


class SegmenterRunner:
    """Runs the U-Net on crops around flagged regions.

    Usage:
        Built by the model loader; ``segment_regions(rgb_image, regions)`` per page.
    """

    def __init__(self, segmenter: nn.Module, crop_size: int, compute_device: torch.device) -> None:
        """Store the model.

        Args:
            segmenter: Trained U-Net (moved to ``compute_device``, eval mode).
            crop_size: Crop side in pixels.
            compute_device: Device.
        """
        self._segmenter = segmenter.eval().to(compute_device)
        self._crop_size = crop_size
        self._compute_device = compute_device

    @torch.inference_mode()
    def predict_crops(self, crop_batch: list[numpy.ndarray]) -> numpy.ndarray:
        """Sigmoid probabilities of a batch of RGB crops.

        Args:
            crop_batch: (S, S, 3) uint8 crops.

        Returns:
            (B, K, S, S) uint8 probabilities scaled to 0..255.
        """
        image_batch = torch.stack([normalise_crop(rgb_crop) for rgb_crop in crop_batch]).to(self._compute_device)
        with torch.autocast(
            self._compute_device.type, dtype=torch.bfloat16, enabled=self._compute_device.type == "cuda"
        ):
            pixel_logits = self._segmenter(image_batch)
        return (torch.sigmoid(pixel_logits.float()) * 255.0 + 0.5).to(torch.uint8).cpu().numpy()

    def segment_regions(self, rgb_image: numpy.ndarray, region_boxes: list[PixelBox]) -> numpy.ndarray:
        """Predict all classes inside the regions; pixels outside every region stay zero.

        Args:
            rgb_image: (H, W, 3) uint8 page.
            region_boxes: Flagged regions.

        Returns:
            (K, H, W) uint8 probabilities scaled to 0..255 (max over overlapping crops).
        """
        page_height, page_width = rgb_image.shape[:2]
        probability_maps: numpy.ndarray | None = None
        all_tiles = region_tiles(region_boxes, self._crop_size)
        for batch_start in range(0, len(all_tiles), TILE_BATCH):
            batch_tiles = all_tiles[batch_start : batch_start + TILE_BATCH]
            crop_batch = [
                padded_crop(rgb_image, row_start, column_start, self._crop_size, PADDING_VALUE)
                for row_start, column_start, _ in batch_tiles
            ]
            crop_probabilities = self.predict_crops(crop_batch)
            if probability_maps is None:
                probability_maps = numpy.zeros((crop_probabilities.shape[1], page_height, page_width), numpy.uint8)
            for (row_start, column_start, region_box), tile_probabilities in zip(
                batch_tiles, crop_probabilities, strict=True
            ):
                paste_region(probability_maps, tile_probabilities, row_start, column_start, region_box)
        return (
            probability_maps if probability_maps is not None else numpy.zeros((1, page_height, page_width), numpy.uint8)
        )


def paste_region(
    probability_maps: numpy.ndarray,
    tile_probabilities: numpy.ndarray,
    row_start: int,
    column_start: int,
    region_box: PixelBox,
) -> None:
    """Max-merge the part of a tile that lies inside its region into the page maps.

    Args:
        probability_maps: (K, H, W) uint8 page maps, mutated.
        tile_probabilities: (K, S, S) uint8 tile output.
        row_start: Tile top row on the page (may be negative).
        column_start: Tile left column on the page (may be negative).
        region_box: Region the tile serves.

    Returns:
        None.
    """
    crop_size = tile_probabilities.shape[1]
    left_edge, top_edge = max(region_box[0], column_start, 0), max(region_box[1], row_start, 0)
    right_edge = min(region_box[2], column_start + crop_size, probability_maps.shape[2])
    bottom_edge = min(region_box[3], row_start + crop_size, probability_maps.shape[1])
    if right_edge <= left_edge or bottom_edge <= top_edge:
        return
    tile_rows = slice(top_edge - row_start, bottom_edge - row_start)
    tile_columns = slice(left_edge - column_start, right_edge - column_start)
    tile_window = tile_probabilities[:, tile_rows, tile_columns]
    page_window = probability_maps[:, top_edge:bottom_edge, left_edge:right_edge]
    numpy.maximum(page_window, tile_window, out=page_window)
