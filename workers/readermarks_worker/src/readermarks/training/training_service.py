"""High-level training actions used by the CLI: feature caching, spotter and segmenter training."""

from pathlib import Path

import torch

from readermarks.datasets.page_inventory import load_real_pages
from readermarks.networks.backbone_wrapper import DinoBackbone
from readermarks.networks.checkpoint_store import SEGMENTER_FILE, SPOTTER_FILE
from readermarks.networks.feature_cache import CacheItem, cache_items, free_backbone_memory
from readermarks.settings.model_settings import ReaderMarkSettings
from readermarks.synthesis.dataset_writer import read_manifest
from readermarks.synthesis.synthesis_service import synthetic_split_directory
from readermarks.training.segmenter_training import SegmenterTrainingResult, train_segmenter
from readermarks.training.spotter_training import SpotterTrainingResult, train_spotter

FEATURES_DIRECTORY = "features"
REAL_CACHE_SPLIT = "real"


def compute_device() -> torch.device:
    """Pick the CUDA device when available.

    Returns:
        ``cuda`` or ``cpu`` device.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def feature_directory(settings: ReaderMarkSettings, split_name: str) -> Path:
    """Directory of cached features for a split and the configured backbone.

    Args:
        settings: Pipeline settings.
        split_name: ``train``, ``validation``, ``test`` or ``real``.

    Returns:
        ``<data>/features/<backbone>/<split>``.
    """
    backbone_name = settings.path_settings.backbone_directory.name
    return (
        settings.path_settings.resolved(settings.path_settings.data_directory)
        / FEATURES_DIRECTORY
        / backbone_name
        / split_name
    )


def synthetic_cache_items(settings: ReaderMarkSettings, split_name: str) -> list[CacheItem]:
    """Cache items of a generated synthetic split.

    Args:
        settings: Pipeline settings.
        split_name: Synthetic split.

    Returns:
        Items with image and mask archive paths.
    """
    split_directory = synthetic_split_directory(settings, split_name)
    return [
        CacheItem(
            manifest_record["sample"],
            split_directory / f"{manifest_record['sample']}.png",
            split_directory / f"{manifest_record['sample']}_masks.npz",
        )
        for manifest_record in read_manifest(split_directory)
    ]


def cache_all_features(settings: ReaderMarkSettings) -> dict[str, int]:
    """Encode all synthetic splits and all real pages with the frozen backbone.

    Usage:
        ``python -m readermarks cache-features``.

    Args:
        settings: Pipeline settings.

    Returns:
        Mapping split -> number of newly encoded items.
    """
    backbone = DinoBackbone(
        settings.path_settings.resolved(settings.path_settings.backbone_directory), compute_device()
    )
    encoded_counts = {}
    for split_name in ("train", "validation", "test"):
        encoded_counts[split_name] = cache_items(
            synthetic_cache_items(settings, split_name),
            backbone,
            settings.spotter_settings,
            feature_directory(settings, split_name),
        )
    real_pages = load_real_pages(settings.path_settings.resolved(settings.path_settings.real_pages_directory))
    real_items = [
        CacheItem(f"page_{real_page.page_number:04d}", real_page.image_path, None) for real_page in real_pages
    ]
    encoded_counts[REAL_CACHE_SPLIT] = cache_items(
        real_items, backbone, settings.spotter_settings, feature_directory(settings, REAL_CACHE_SPLIT)
    )
    free_backbone_memory()
    return encoded_counts


def run_spotter_training(settings: ReaderMarkSettings) -> SpotterTrainingResult:
    """Train the spotter head on the cached synthetic features.

    Usage:
        ``python -m readermarks train-spotter``.

    Args:
        settings: Pipeline settings.

    Returns:
        Training summary.
    """
    backbone_directory = settings.path_settings.resolved(settings.path_settings.backbone_directory)
    from transformers import AutoConfig  # local import: only the config is needed here

    # B615: the config is read from a local directory only (local_files_only=True), never downloaded.
    backbone_config = AutoConfig.from_pretrained(str(backbone_directory), local_files_only=True)  # nosec B615
    feature_channels = int(backbone_config.hidden_size)
    return train_spotter(
        feature_directory(settings, "train"),
        feature_directory(settings, "validation"),
        feature_channels,
        settings.spotter_settings,
        settings.path_settings.resolved(settings.path_settings.checkpoint_directory) / SPOTTER_FILE,
        backbone_directory.name,
        compute_device(),
    )


def run_segmenter_training(settings: ReaderMarkSettings) -> SegmenterTrainingResult:
    """Train the U-Net segmenter on native-resolution synthetic crops.

    Usage:
        ``python -m readermarks train-segmenter``.

    Args:
        settings: Pipeline settings.

    Returns:
        Training summary.
    """
    return train_segmenter(
        synthetic_split_directory(settings, "train"),
        synthetic_split_directory(settings, "validation"),
        settings.segmenter,
        settings.path_settings.resolved(settings.path_settings.encoder_directory),
        settings.path_settings.resolved(settings.path_settings.checkpoint_directory) / SEGMENTER_FILE,
        compute_device(),
    )
