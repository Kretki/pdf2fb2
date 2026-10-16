"""Stage 2 segmenter: U-Net with a ConvNeXt-Tiny encoder (segmentation_models_pytorch).

ImageNet weights come from the local timm checkpoint; nothing is downloaded.
"""

import re
from pathlib import Path

import segmentation_models_pytorch
import torch
from safetensors.torch import load_file
from torch import nn

ENCODER_WEIGHTS_FILE = "model.safetensors"
MODULE_INDEX_PATTERN = re.compile(r"^(stem|stages)\.(\d+)")


def remap_timm_keys(timm_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename a classification checkpoint's keys to the timm ``features_only`` layout.

    Args:
        timm_state: State dict of ``convnext_tiny`` (keys like ``stages.0.blocks.0...``).

    Returns:
        State dict with ``stages_0...`` / ``stem_0...`` keys and no classifier head.
    """
    remapped_state = {}
    for parameter_name, parameter_value in timm_state.items():
        if parameter_name.startswith("head."):
            continue
        remapped_state[MODULE_INDEX_PATTERN.sub(r"\1_\2", parameter_name)] = parameter_value
    return remapped_state


def build_segmenter(encoder_name: str, class_count: int, encoder_directory: Path | None) -> nn.Module:
    """Create the U-Net and optionally load ImageNet encoder weights from a local directory.

    Usage:
        Training passes ``encoder_directory``; inference passes None and loads a full checkpoint.

    Args:
        encoder_name: smp encoder name, e.g. ``tu-convnext_tiny``.
        class_count: Output channels (multi-label sigmoid logits).
        encoder_directory: Directory with the timm ``model.safetensors``, or None.

    Returns:
        The U-Net module.

    Raises:
        RuntimeError: If the encoder weights do not match the encoder.
    """
    segmenter = segmentation_models_pytorch.Unet(encoder_name=encoder_name, encoder_weights=None, classes=class_count)
    if encoder_directory is not None:
        encoder_state = remap_timm_keys(load_file(str(encoder_directory / ENCODER_WEIGHTS_FILE)))
        load_report = segmenter.encoder.model.load_state_dict(encoder_state, strict=False)
        if load_report.missing_keys:
            raise RuntimeError(
                f"Encoder weights miss {len(load_report.missing_keys)} keys, e.g. {load_report.missing_keys[:3]}."
            )
    return segmenter
