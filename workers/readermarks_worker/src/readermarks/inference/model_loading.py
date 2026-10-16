"""Load the trained spotter/segmenter and the tuned thresholds into ready-to-use runners."""

from dataclasses import dataclass
from pathlib import Path

import torch

from readermarks.inference.page_cleaner import PageCleaner
from readermarks.inference.segmenter_inference import SegmenterRunner
from readermarks.inference.spotter_inference import SpotterRunner
from readermarks.networks.backbone_wrapper import DinoBackbone
from readermarks.networks.checkpoint_store import SEGMENTER_FILE, SPOTTER_FILE, load_checkpoint, load_thresholds
from readermarks.networks.segmenter_network import build_segmenter
from readermarks.networks.spotter_head import SpotterHead
from readermarks.settings.model_settings import CleaningSettings, ReaderMarkSettings


def effective_cleaning_settings(settings: ReaderMarkSettings) -> CleaningSettings:
    """Cleaning settings with the tuned ``thresholds.json`` values applied on top.

    Args:
        settings: Pipeline settings.

    Returns:
        Validated cleaning settings.
    """
    tuned_values = load_thresholds(settings.path_settings.resolved(settings.path_settings.checkpoint_directory))
    return CleaningSettings.model_validate({**settings.cleaning.model_dump(), **tuned_values})


def load_spotter_runner(settings: ReaderMarkSettings, compute_device: torch.device) -> SpotterRunner:
    """Build the frozen backbone and the trained head.

    Args:
        settings: Pipeline settings.
        compute_device: Device.

    Returns:
        Spotter runner.

    Raises:
        FileNotFoundError: If the backbone or the spotter checkpoint is missing.
    """
    head_state, head_meta = load_checkpoint(
        settings.path_settings.resolved(settings.path_settings.checkpoint_directory) / SPOTTER_FILE
    )
    backbone_directory = settings.path_settings.resolved(settings.path_settings.backbone_directory)
    if backbone_directory.name != head_meta["backbone"]:
        raise ValueError(
            f"Spotter was trained on {head_meta['backbone']}, configured backbone is {backbone_directory.name}."
        )
    spotter_head = SpotterHead(
        head_meta["feature_channels"],
        head_meta["hidden_channels"],
        head_meta["convolution_layers"],
        head_meta["class_count"],
    )
    spotter_head.load_state_dict(head_state)
    return SpotterRunner(
        DinoBackbone(backbone_directory, compute_device), spotter_head, settings.spotter_settings, compute_device
    )


def load_segmenter_runner(settings: ReaderMarkSettings, compute_device: torch.device) -> SegmenterRunner:
    """Build the trained U-Net.

    Args:
        settings: Pipeline settings.
        compute_device: Device.

    Returns:
        Segmenter runner.

    Raises:
        FileNotFoundError: If the segmenter checkpoint is missing.
    """
    checkpoint_path: Path = (
        settings.path_settings.resolved(settings.path_settings.checkpoint_directory) / SEGMENTER_FILE
    )
    segmenter_state, segmenter_meta = load_checkpoint(checkpoint_path)
    segmenter = build_segmenter(segmenter_meta["encoder_name"], segmenter_meta["class_count"], None)
    segmenter.load_state_dict(segmenter_state)
    return SegmenterRunner(segmenter, settings.segmenter.crop_size, compute_device)


@dataclass(frozen=True)
class LoadedModels:
    """Runners and cleaner built from the trained checkpoints.

    Attributes:
        spotter_runner: Stage 1 runner.
        segmenter_runner: Stage 2 runner.
        page_cleaner: Full pipeline with the effective thresholds.
    """

    spotter_runner: SpotterRunner
    segmenter_runner: SegmenterRunner
    page_cleaner: PageCleaner


def load_models(settings: ReaderMarkSettings, compute_device: torch.device) -> LoadedModels:
    """Load both stages and build the cleaner.

    Usage:
        Called once per process by the CLI commands and the pdf2fb2 worker.

    Args:
        settings: Pipeline settings.
        compute_device: Device.

    Returns:
        Loaded models.

    Raises:
        FileNotFoundError: If a checkpoint or the backbone is missing.
    """
    spotter_runner = load_spotter_runner(settings, compute_device)
    segmenter_runner = load_segmenter_runner(settings, compute_device)
    return LoadedModels(
        spotter_runner,
        segmenter_runner,
        PageCleaner(spotter_runner, segmenter_runner, effective_cleaning_settings(settings)),
    )
