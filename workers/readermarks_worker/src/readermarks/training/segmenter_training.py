"""Train the Stage 2 U-Net segmenter on synthetic crops (BCE + Dice, bf16, STAGE_CLEAN.md §6)."""

import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from readermarks.datasets.segmenter_dataset import SegmenterCropDataset
from readermarks.networks.checkpoint_store import save_checkpoint
from readermarks.networks.segmenter_network import build_segmenter
from readermarks.settings.class_catalog import SEGMENTER_CLASS_COUNT, segmenter_channel
from readermarks.settings.model_settings import SegmenterSettings
from readermarks.training.loss_functions import bce_dice

# Positive-pixel weights: marks are rare and thin, ink is common.
CHANNEL_POSITIVE_WEIGHTS = (3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 1.0, 3.0)


@dataclass(frozen=True)
class SegmenterTrainingResult:
    """Summary of a segmenter training run.

    Attributes:
        checkpoint_path: Saved model.
        best_mark_iou: Best ``mark_any`` IoU on the validation crops.
        peak_memory_gigabytes: Peak CUDA memory.
        training_seconds: Wall time.
    """

    checkpoint_path: Path
    best_mark_iou: float
    peak_memory_gigabytes: float
    training_seconds: float


def endless_batches(data_loader: DataLoader):
    """Yield batches forever, restarting the loader each epoch.

    Args:
        data_loader: Training loader.

    Yields:
        ``(images, targets)`` batches.
    """
    while True:
        yield from data_loader


@torch.inference_mode()
def validation_mark_iou(
    segmenter: torch.nn.Module, validation_loader: DataLoader, compute_device: torch.device
) -> float:
    """IoU of ``mark_any`` (threshold 0.5) over all validation crops.

    Args:
        segmenter: Model.
        validation_loader: Deterministic crops.
        compute_device: Device.

    Returns:
        IoU 0..1.
    """
    segmenter.eval()
    intersection_total, union_total = 0.0, 0.0
    mark_channel = segmenter_channel("mark_any")
    for image_batch, target_batch in validation_loader:
        with torch.autocast(compute_device.type, dtype=torch.bfloat16, enabled=compute_device.type == "cuda"):
            pixel_logits = segmenter(image_batch.to(compute_device))
        predicted = pixel_logits[:, mark_channel].float() > 0
        expected = target_batch[:, mark_channel].to(compute_device) > 0.5
        intersection_total += float((predicted & expected).sum())
        union_total += float((predicted | expected).sum())
    segmenter.train()
    return intersection_total / max(1.0, union_total)


def segmenter_loaders(
    training_directory: Path, validation_directory: Path, segmenter_settings: SegmenterSettings
) -> tuple[DataLoader, DataLoader]:
    """Random training crops and deterministic validation crops.

    Args:
        training_directory: Synthetic training split.
        validation_directory: Synthetic validation split.
        segmenter_settings: Crop size, batch size, loader workers.

    Returns:
        ``(training_loader, validation_loader)``.
    """
    crop_size, mark_fraction = segmenter_settings.crop_size, segmenter_settings.mark_crop_fraction
    training_loader = DataLoader(
        SegmenterCropDataset(training_directory, crop_size, mark_fraction, training=True),
        batch_size=segmenter_settings.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=segmenter_settings.loader_workers,
        persistent_workers=True,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        SegmenterCropDataset(validation_directory, crop_size, mark_fraction, training=False),
        batch_size=segmenter_settings.batch_size,
        num_workers=4,
        pin_memory=True,
    )
    return training_loader, validation_loader


def optimisation_step(
    segmenter: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_pair: tuple,
    positive_weights: torch.Tensor,
    compute_device: torch.device,
) -> float:
    """One bf16 forward/backward/update step.

    Args:
        segmenter: Model in training mode.
        optimizer: Optimizer.
        batch_pair: ``(images, targets)`` batch.
        positive_weights: (K,) BCE positive weights on the device.
        compute_device: Device.

    Returns:
        The batch loss.
    """
    image_batch, target_batch = batch_pair
    with torch.autocast(compute_device.type, dtype=torch.bfloat16, enabled=compute_device.type == "cuda"):
        pixel_logits = segmenter(image_batch.to(compute_device, non_blocking=True))
    batch_loss = bce_dice(pixel_logits.float(), target_batch.to(compute_device, non_blocking=True), positive_weights)
    optimizer.zero_grad(set_to_none=True)
    batch_loss.backward()
    optimizer.step()
    return batch_loss.item()


def train_segmenter(
    training_directory: Path,
    validation_directory: Path,
    segmenter_settings: SegmenterSettings,
    encoder_directory: Path,
    checkpoint_path: Path,
    compute_device: torch.device,
) -> SegmenterTrainingResult:
    """Train for ``step_count`` steps, keeping the checkpoint with the best validation IoU.

    Usage:
        ``python -m readermarks train-segmenter``.

    Args:
        training_directory: Synthetic training split.
        validation_directory: Synthetic validation split.
        segmenter_settings: Architecture and optimisation settings.
        encoder_directory: Local ImageNet ConvNeXt-Tiny weights.
        checkpoint_path: Where the best model is saved.
        compute_device: Training device.

    Returns:
        Training summary.
    """
    start_time = time.perf_counter()
    segmenter = build_segmenter(segmenter_settings.encoder_name, SEGMENTER_CLASS_COUNT, encoder_directory)
    segmenter.to(compute_device).train()
    training_loader, validation_loader = segmenter_loaders(training_directory, validation_directory, segmenter_settings)
    optimizer = torch.optim.AdamW(
        segmenter.parameters(), lr=segmenter_settings.learning_rate, weight_decay=segmenter_settings.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=segmenter_settings.learning_rate, total_steps=segmenter_settings.step_count, pct_start=0.05
    )
    positive_weights = torch.tensor(CHANNEL_POSITIVE_WEIGHTS, device=compute_device)
    metadata = {"encoder_name": segmenter_settings.encoder_name, "class_count": SEGMENTER_CLASS_COUNT}
    best_iou = -1.0
    batch_stream = endless_batches(training_loader)
    for step_index in range(1, segmenter_settings.step_count + 1):
        batch_loss = optimisation_step(segmenter, optimizer, next(batch_stream), positive_weights, compute_device)
        scheduler.step()
        if step_index % segmenter_settings.validation_interval == 0 or step_index == segmenter_settings.step_count:
            mark_iou = validation_mark_iou(segmenter, validation_loader, compute_device)
            print(f"segmenter step {step_index}: loss {batch_loss:.4f} val mark IoU {mark_iou:.4f}", flush=True)
            if mark_iou > best_iou:
                best_iou = mark_iou
                save_checkpoint(checkpoint_path, segmenter, metadata)
    peak_memory = torch.cuda.max_memory_allocated() / 1e9 if compute_device.type == "cuda" else 0.0
    return SegmenterTrainingResult(checkpoint_path, best_iou, peak_memory, time.perf_counter() - start_time)
