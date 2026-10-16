"""Losses: masked BCE + focal (spotter, heavy imbalance) and BCE + Dice (segmenter)."""

import torch
import torch.nn.functional as functional


def masked_bce_focal(
    patch_logits: torch.Tensor,
    patch_labels: torch.Tensor,
    valid_mask: torch.Tensor,
    positive_weights: torch.Tensor,
    focal_gamma: float,
    focal_weight: float,
) -> torch.Tensor:
    """Class-weighted BCE plus focal loss over valid patches.

    Args:
        patch_logits: (B, K, R, C) logits.
        patch_labels: (B, K, R, C) 0/1 targets.
        valid_mask: (B, 1, R, C) 1 inside the page.
        positive_weights: (K,) weight of positive patches per class (imbalance handling).
        focal_gamma: Focusing parameter (>= 0).
        focal_weight: Weight of the focal term.

    Returns:
        Scalar loss.
    """
    weight_map = positive_weights.view(1, -1, 1, 1)
    bce_values = functional.binary_cross_entropy_with_logits(patch_logits, patch_labels, reduction="none")
    probabilities = torch.sigmoid(patch_logits)
    target_probability = probabilities * patch_labels + (1 - probabilities) * (1 - patch_labels)
    focal_map = (1 - target_probability) ** focal_gamma * bce_values
    weighted_bce = bce_values * (1 + (weight_map - 1) * patch_labels)
    combined_map = (weighted_bce + focal_weight * focal_map) * valid_mask
    return combined_map.sum() / (valid_mask.sum() * patch_logits.shape[1]).clamp_min(1.0)


def bce_dice(pixel_logits: torch.Tensor, pixel_labels: torch.Tensor, positive_weights: torch.Tensor) -> torch.Tensor:
    """Weighted BCE plus soft Dice per channel (multi-label).

    Args:
        pixel_logits: (B, K, H, W) logits.
        pixel_labels: (B, K, H, W) 0/1 targets.
        positive_weights: (K,) positive weight per channel for the BCE.

    Returns:
        Scalar loss.
    """
    bce_term = functional.binary_cross_entropy_with_logits(
        pixel_logits, pixel_labels, pos_weight=positive_weights.view(1, -1, 1, 1)
    )
    probabilities = torch.sigmoid(pixel_logits)
    intersection = (probabilities * pixel_labels).sum(dim=(0, 2, 3))
    denominator = probabilities.sum(dim=(0, 2, 3)) + pixel_labels.sum(dim=(0, 2, 3))
    dice_term = 1 - (2 * intersection + 1.0) / (denominator + 1.0)
    return bce_term + dice_term.mean()
