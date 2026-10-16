"""Train the Stage 1 spotter head on cached features (STAGE_CLEAN.md §6, stage 1)."""

import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from readermarks.datasets.spotter_dataset import CachedSpotterDataset, collate_padded
from readermarks.networks.checkpoint_store import save_checkpoint
from readermarks.networks.spotter_head import SpotterHead
from readermarks.settings.class_catalog import SPOTTER_CLASS_COUNT
from readermarks.settings.model_settings import SpotterSettings
from readermarks.training.loss_functions import masked_bce_focal


@dataclass(frozen=True)
class SpotterTrainingResult:
    """Summary of a spotter training run.

    Attributes:
        checkpoint_path: Saved head.
        best_validation_loss: Lowest validation loss.
        training_seconds: Wall time.
    """

    checkpoint_path: Path
    best_validation_loss: float
    training_seconds: float


def positive_class_weights(training_data: CachedSpotterDataset, maximum_weight: float) -> torch.Tensor:
    """Weight of positive patches per class: sqrt(negatives/positives), capped.

    Args:
        training_data: Training dataset.
        maximum_weight: Upper bound of a weight.

    Returns:
        (K,) float tensor.
    """
    positive_counts = torch.zeros(SPOTTER_CLASS_COUNT)
    total_patches = 0
    for sample_index in range(0, len(training_data), max(1, len(training_data) // 300)):
        _, patch_labels, _ = training_data[sample_index]
        positive_counts += patch_labels.sum(dim=(1, 2))
        total_patches += patch_labels.shape[1] * patch_labels.shape[2]
    class_ratio = (total_patches - positive_counts) / positive_counts.clamp_min(1.0)
    return class_ratio.sqrt().clamp(1.0, maximum_weight)


def run_epoch(
    spotter_head: SpotterHead,
    data_loader: DataLoader,
    positive_weights: torch.Tensor,
    spotter_settings: SpotterSettings,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    """One pass over a loader; trains when an optimizer is given, else evaluates.

    Args:
        spotter_head: Head being trained.
        data_loader: Loader with ``collate_padded`` batches.
        positive_weights: (K,) class weights on the head's device.
        spotter_settings: Loss parameters.
        optimizer: Optimizer, or None for evaluation.

    Returns:
        Mean loss of the epoch.
    """
    compute_device = positive_weights.device
    spotter_head.train(optimizer is not None)
    loss_total, batch_count = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for feature_batch, label_batch, valid_mask, page_labels in data_loader:
            feature_batch, label_batch = feature_batch.to(compute_device), label_batch.to(compute_device)
            valid_mask, page_labels = valid_mask.to(compute_device), page_labels.to(compute_device)
            with torch.autocast(compute_device.type, dtype=torch.bfloat16, enabled=compute_device.type == "cuda"):
                patch_logits, page_logit = spotter_head(feature_batch, valid_mask)
            batch_loss = masked_bce_focal(
                patch_logits.float(),
                label_batch,
                valid_mask,
                positive_weights,
                spotter_settings.focal_gamma,
                spotter_settings.focal_weight,
            )
            batch_loss = batch_loss + spotter_settings.page_loss_weight * functional.binary_cross_entropy_with_logits(
                page_logit.float(), page_labels
            )
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                batch_loss.backward()
                optimizer.step()
            loss_total += batch_loss.item()
            batch_count += 1
    return loss_total / max(1, batch_count)


def spotter_loaders(training_directory: Path, validation_directory: Path, batch_size: int) -> tuple:
    """Training (shuffled, flipped) and validation loaders over cached features.

    Args:
        training_directory: Cached training features.
        validation_directory: Cached validation features.
        batch_size: Batch size.

    Returns:
        ``(training_data, training_loader, validation_loader)``.
    """
    training_data = CachedSpotterDataset(training_directory, horizontal_flip=True)
    training_loader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_padded,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        CachedSpotterDataset(validation_directory),
        batch_size=batch_size,
        collate_fn=collate_padded,
        num_workers=2,
        pin_memory=True,
    )
    return training_data, training_loader, validation_loader


def spotter_metadata(backbone_name: str, feature_channels: int, spotter_settings: SpotterSettings) -> dict:
    """Architecture parameters stored with the head checkpoint.

    Args:
        backbone_name: Backbone directory name.
        feature_channels: Backbone token dimension.
        spotter_settings: Head settings.

    Returns:
        JSON-compatible metadata.
    """
    return {
        "backbone": backbone_name,
        "feature_channels": feature_channels,
        "hidden_channels": spotter_settings.hidden_channels,
        "convolution_layers": spotter_settings.convolution_layers,
        "class_count": SPOTTER_CLASS_COUNT,
    }


def train_spotter(
    training_directory: Path,
    validation_directory: Path,
    feature_channels: int,
    spotter_settings: SpotterSettings,
    checkpoint_path: Path,
    backbone_name: str,
    compute_device: torch.device,
) -> SpotterTrainingResult:
    """Train the head and keep the checkpoint with the lowest validation loss.

    Usage:
        ``python -m readermarks train-spotter`` after ``cache-features``.

    Args:
        training_directory: Cached training features.
        validation_directory: Cached validation features.
        feature_channels: Backbone token dimension.
        spotter_settings: Architecture and optimisation settings.
        checkpoint_path: Where the best head is saved.
        backbone_name: Backbone directory name stored in the checkpoint metadata.
        compute_device: Training device.

    Returns:
        Training summary.
    """
    start_time = time.perf_counter()
    training_data, training_loader, validation_loader = spotter_loaders(
        training_directory, validation_directory, spotter_settings.batch_size
    )
    spotter_head = SpotterHead(
        feature_channels, spotter_settings.hidden_channels, spotter_settings.convolution_layers, SPOTTER_CLASS_COUNT
    ).to(compute_device)
    positive_weights = positive_class_weights(training_data, spotter_settings.maximum_positive_weight)
    positive_weights = positive_weights.to(compute_device)
    optimizer = torch.optim.AdamW(
        spotter_head.parameters(), lr=spotter_settings.learning_rate, weight_decay=spotter_settings.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=spotter_settings.epoch_count)
    metadata = spotter_metadata(backbone_name, feature_channels, spotter_settings)
    best_loss = float("inf")
    for _ in range(spotter_settings.epoch_count):
        run_epoch(spotter_head, training_loader, positive_weights, spotter_settings, optimizer)
        scheduler.step()
        validation_loss = run_epoch(spotter_head, validation_loader, positive_weights, spotter_settings, None)
        if validation_loss < best_loss:
            best_loss = validation_loss
            save_checkpoint(checkpoint_path, spotter_head, metadata)
    return SpotterTrainingResult(checkpoint_path, best_loss, time.perf_counter() - start_time)
