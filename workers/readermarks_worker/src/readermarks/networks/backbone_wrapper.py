"""Frozen ViT backbone (DINOv2 by default) that turns a page into a 2-D grid of patch tokens.

Weights are loaded from a local directory only (``HF_HUB_OFFLINE=1``, ``local_files_only``);
any Hugging Face DINOv2 checkpoint directory (small/base/...) can be configured.
"""

import os
from pathlib import Path
from typing import Protocol

import numpy
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PatchBackbone(Protocol):
    """Anything that maps an image batch to patch-token grids.

    Usage:
        Implemented by ``DinoBackbone``; the spotter depends only on this protocol.
    """

    feature_channels: int

    def extract_patch_grid(self, image_batch: torch.Tensor) -> torch.Tensor:
        """Encode images whose sides are multiples of the patch size.

        Args:
            image_batch: (B, 3, H, W) float tensor, ImageNet-normalised.

        Returns:
            (B, feature_channels, H/patch, W/patch) tensor.
        """
        ...


class DinoBackbone:
    """Frozen DINOv2 encoder from a local Hugging Face checkpoint directory.

    Usage:
        ``DinoBackbone(Path("models/dinov2-small"), torch.device("cuda"))``.
    """

    def __init__(self, checkpoint_directory: Path, compute_device: torch.device) -> None:
        """Load the weights (offline) and freeze them.

        Args:
            checkpoint_directory: Directory with ``config.json`` and ``model.safetensors``.
            compute_device: Device to run on.

        Raises:
            FileNotFoundError: If the directory has no config.
        """
        if not (checkpoint_directory / "config.json").is_file():
            raise FileNotFoundError(f"No DINOv2 checkpoint in {checkpoint_directory}.")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from transformers import Dinov2Model  # local import: heavy, only needed when a backbone is built

        # B615: weights are loaded from a local directory only (local_files_only=True), never downloaded.
        self._vision_encoder = Dinov2Model.from_pretrained(str(checkpoint_directory), local_files_only=True)  # nosec B615
        self._vision_encoder.eval().requires_grad_(False).to(compute_device)
        self._compute_device = compute_device
        self.feature_channels = int(self._vision_encoder.config.hidden_size)
        self.patch_size = int(self._vision_encoder.config.patch_size)

    @torch.inference_mode()
    def extract_patch_grid(self, image_batch: torch.Tensor) -> torch.Tensor:
        """Encode a normalised batch into a patch-token grid (bf16 autocast on CUDA).

        Args:
            image_batch: (B, 3, H, W) float tensor; H and W are multiples of ``patch_size``.

        Returns:
            (B, feature_channels, H/patch, W/patch) float32 tensor.
        """
        grid_rows = image_batch.shape[2] // self.patch_size
        grid_columns = image_batch.shape[3] // self.patch_size
        with torch.autocast(
            self._compute_device.type, dtype=torch.bfloat16, enabled=self._compute_device.type == "cuda"
        ):
            hidden_states = self._vision_encoder(pixel_values=image_batch.to(self._compute_device)).last_hidden_state
        register_count = hidden_states.shape[1] - grid_rows * grid_columns
        patch_tokens = hidden_states[:, register_count:].float()
        return patch_tokens.transpose(1, 2).reshape(image_batch.shape[0], -1, grid_rows, grid_columns)


def normalise_image(prepared_image: numpy.ndarray) -> torch.Tensor:
    """Convert a prepared uint8 RGB page to a normalised (1, 3, H, W) tensor.

    Args:
        prepared_image: Output of ``prepare_spotter_image``.

    Returns:
        Float32 tensor on the CPU.
    """
    image_tensor = torch.from_numpy(prepared_image).permute(2, 0, 1).float().div_(255.0)
    mean_tensor = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std_tensor = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return ((image_tensor - mean_tensor) / std_tensor).unsqueeze(0)
