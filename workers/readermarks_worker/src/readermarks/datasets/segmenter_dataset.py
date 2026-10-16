"""Native-resolution crops of synthetic pages with multi-label pixel targets for the segmenter."""

from pathlib import Path

import cv2
import numpy
import torch
from torch.utils.data import Dataset

from readermarks.datasets.image_files import read_rgb_image
from readermarks.datasets.mask_archive import read_mask_archive
from readermarks.networks.backbone_wrapper import IMAGENET_MEAN, IMAGENET_STD
from readermarks.settings.class_catalog import MARK_CLASS_COUNT
from readermarks.synthesis.dataset_writer import read_manifest

PADDING_VALUE = 255


def segmenter_targets(class_masks: numpy.ndarray) -> numpy.ndarray:
    """Prepend the ``mark_any`` channel to the spotter-ordered class masks.

    Args:
        class_masks: (9, H, W) 0/1 masks in ``SPOTTER_CLASSES`` order.

    Returns:
        (10, H, W) uint8 masks in ``SEGMENTER_CLASSES`` order.
    """
    mark_any = class_masks[:MARK_CLASS_COUNT].max(axis=0, keepdims=True)
    return numpy.concatenate([mark_any, class_masks], axis=0)


def normalise_crop(rgb_crop: numpy.ndarray) -> torch.Tensor:
    """ImageNet-normalise an RGB uint8 crop.

    Args:
        rgb_crop: (h, w, 3) uint8.

    Returns:
        (3, h, w) float32 tensor.
    """
    crop_tensor = torch.from_numpy(numpy.ascontiguousarray(rgb_crop)).permute(2, 0, 1).float().div_(255.0)
    return (crop_tensor - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)) / torch.tensor(IMAGENET_STD).view(3, 1, 1)


def choose_crop_origin(
    target_masks: numpy.ndarray, crop_size: int, prefer_marks: bool, random_source: numpy.random.Generator
) -> tuple[int, int]:
    """Top-left corner of a crop, centered on a random mark pixel when asked.

    Args:
        target_masks: (K, H, W) targets (channel 0 = mark_any).
        crop_size: Crop side in pixels.
        prefer_marks: Center on a mark pixel if the page has any.
        random_source: Random generator.

    Returns:
        ``(row_start, column_start)``; may be negative when the page is smaller than the crop.
    """
    page_height, page_width = target_masks.shape[1:]
    mark_rows, mark_columns = numpy.nonzero(target_masks[0]) if prefer_marks else (numpy.array([]), numpy.array([]))
    if mark_rows.size:
        pick_index = int(random_source.integers(mark_rows.size))
        center_row = mark_rows[pick_index] + int(random_source.integers(-crop_size // 4, crop_size // 4 + 1))
        center_column = mark_columns[pick_index] + int(random_source.integers(-crop_size // 4, crop_size // 4 + 1))
        row_start, column_start = center_row - crop_size // 2, center_column - crop_size // 2
    else:
        row_start = int(random_source.integers(0, max(1, page_height - crop_size + 1)))
        column_start = int(random_source.integers(0, max(1, page_width - crop_size + 1)))
    row_start = int(numpy.clip(row_start, min(0, page_height - crop_size), max(0, page_height - crop_size)))
    column_start = int(numpy.clip(column_start, min(0, page_width - crop_size), max(0, page_width - crop_size)))
    return row_start, column_start


def padded_crop(
    source_array: numpy.ndarray, row_start: int, column_start: int, crop_size: int, padding_value: int
) -> numpy.ndarray:
    """Cut a square crop (channels last), padding outside the source.

    Args:
        source_array: (H, W) or (H, W, C) array.
        row_start: Top row (may be negative).
        column_start: Left column (may be negative).
        crop_size: Crop side.
        padding_value: Fill value outside the source.

    Returns:
        (crop_size, crop_size[, C]) array of the same dtype.
    """
    crop_shape = (crop_size, crop_size) + source_array.shape[2:]
    crop_array = numpy.full(crop_shape, padding_value, source_array.dtype)
    source_rows = slice(max(0, row_start), min(source_array.shape[0], row_start + crop_size))
    source_columns = slice(max(0, column_start), min(source_array.shape[1], column_start + crop_size))
    crop_array[
        source_rows.start - row_start : source_rows.stop - row_start,
        source_columns.start - column_start : source_columns.stop - column_start,
    ] = source_array[source_rows, source_columns]
    return crop_array


class SegmenterCropDataset(Dataset):
    """Random (training) or deterministic (validation) crops of a synthetic split.

    Usage:
        ``SegmenterCropDataset(split_directory, 768, 0.8, training=True)``.
    """

    def __init__(self, split_directory: Path, crop_size: int, mark_crop_fraction: float, training: bool) -> None:
        """Index the split.

        Args:
            split_directory: Generated synthetic split.
            crop_size: Crop side in pixels.
            mark_crop_fraction: Share of crops centered on marks, 0..1.
            training: Random crops + scale jitter when True; seeded crops otherwise.
        """
        self._split_directory = split_directory
        self._sample_names = [manifest_record["sample"] for manifest_record in read_manifest(split_directory)]
        self._crop_size = crop_size
        self._mark_crop_fraction = mark_crop_fraction
        self._training = training

    def __len__(self) -> int:
        """Return the number of pages.

        Returns:
            Page count.
        """
        return len(self._sample_names)

    def __getitem__(self, sample_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load a page and cut one crop.

        Args:
            sample_index: 0..len-1.

        Returns:
            ``(image (3, S, S) float32, targets (10, S, S) float32)``.
        """
        random_source = numpy.random.default_rng(None if self._training else sample_index)
        sample_name = self._sample_names[sample_index]
        page_image = read_rgb_image(self._split_directory / f"{sample_name}.png")
        target_masks = segmenter_targets(read_mask_archive(self._split_directory / f"{sample_name}_masks.npz"))
        if self._training and random_source.random() < 0.5:
            scale_factor = random_source.uniform(0.85, 1.2)
            page_image = cv2.resize(page_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
            target_masks = numpy.stack(
                [
                    cv2.resize(target_mask, (page_image.shape[1], page_image.shape[0]), interpolation=cv2.INTER_NEAREST)
                    for target_mask in target_masks
                ]
            )
        prefer_marks = random_source.random() < self._mark_crop_fraction
        row_start, column_start = choose_crop_origin(target_masks, self._crop_size, prefer_marks, random_source)
        image_crop = padded_crop(page_image, row_start, column_start, self._crop_size, PADDING_VALUE)
        target_crop = padded_crop(target_masks.transpose(1, 2, 0), row_start, column_start, self._crop_size, 0)
        return normalise_crop(image_crop), torch.from_numpy(target_crop.transpose(2, 0, 1).astype(numpy.float32))
