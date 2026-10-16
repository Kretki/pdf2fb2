"""Precompute frozen-backbone patch features (fp16 ``.npy``) and patch labels once per sample.

Layout: ``<data>/features/<backbone>/<split>/<sample>_features.npy`` (D, rows, columns) fp16 and
``<sample>_labels.npy`` (classes, rows, columns) uint8. Training then only touches the head.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy
import torch

from readermarks.datasets.image_files import read_rgb_image
from readermarks.datasets.mask_archive import read_mask_archive
from readermarks.datasets.patch_grid import compute_geometry, masks_to_patch_labels, prepare_spotter_image
from readermarks.networks.backbone_wrapper import PatchBackbone, normalise_image
from readermarks.settings.model_settings import SpotterSettings

FEATURES_SUFFIX = "_features.npy"
LABELS_SUFFIX = "_labels.npy"


@dataclass(frozen=True)
class CacheItem:
    """One image to encode.

    Attributes:
        item_name: Output file stem.
        image_path: Page image.
        masks_path: Mask archive (synthetic samples) or None (real pages without pixel labels).
    """

    item_name: str
    image_path: Path
    masks_path: Path | None


def encode_page(rgb_image: numpy.ndarray, backbone: PatchBackbone, spotter_settings: SpotterSettings) -> numpy.ndarray:
    """Run the frozen backbone on one page.

    Args:
        rgb_image: Page (H, W, 3) uint8.
        backbone: Patch backbone.
        spotter_settings: Input size and patch size.

    Returns:
        (D, rows, columns) float16 features.
    """
    grid_geometry = compute_geometry(
        rgb_image.shape[0], rgb_image.shape[1], spotter_settings.input_long_side, spotter_settings.patch_size
    )
    image_tensor = normalise_image(prepare_spotter_image(rgb_image, grid_geometry))
    return backbone.extract_patch_grid(image_tensor)[0].cpu().numpy().astype(numpy.float16)


def cache_items(
    cache_items_list: Iterable[CacheItem],
    backbone: PatchBackbone,
    spotter_settings: SpotterSettings,
    output_directory: Path,
) -> int:
    """Encode every item (skipping already cached ones) and store features and labels.

    Usage:
        Called by the ``cache-features`` command for synthetic splits and real pages.

    Args:
        cache_items_list: Items to encode.
        backbone: Patch backbone.
        spotter_settings: Input size, patch size and patch-label fraction.
        output_directory: Directory of the cached split.

    Returns:
        Number of newly encoded items.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    encoded_count = 0
    for cache_item in cache_items_list:
        features_path = output_directory / f"{cache_item.item_name}{FEATURES_SUFFIX}"
        if features_path.is_file():
            continue
        rgb_image = read_rgb_image(cache_item.image_path)
        numpy.save(features_path, encode_page(rgb_image, backbone, spotter_settings))
        if cache_item.masks_path is not None:
            grid_geometry = compute_geometry(
                rgb_image.shape[0], rgb_image.shape[1], spotter_settings.input_long_side, spotter_settings.patch_size
            )
            patch_labels = masks_to_patch_labels(
                read_mask_archive(cache_item.masks_path), grid_geometry, spotter_settings.patch_label_fraction
            )
            numpy.save(output_directory / f"{cache_item.item_name}{LABELS_SUFFIX}", patch_labels)
        encoded_count += 1
    return encoded_count


def free_backbone_memory() -> None:
    """Release cached CUDA memory after encoding.

    Returns:
        None.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
