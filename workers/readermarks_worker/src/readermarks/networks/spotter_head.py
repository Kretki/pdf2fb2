"""Stage 1 spotter head: small conv stack over the patch-token grid plus a page-level logit.

The 3x3 convolutions make neighbouring patches agree (an underline reads as a chain), which
replaces the CRF of Pandey & Harit.
"""

import torch
from torch import nn


class SpotterHead(nn.Module):
    """Patch classifier over frozen backbone features.

    Usage:
        ``patch_logits, page_logit = head(features, valid_mask)``.
    """

    def __init__(self, feature_channels: int, hidden_channels: int, convolution_layers: int, class_count: int) -> None:
        """Build the layers.

        Args:
            feature_channels: Backbone token dimension (384 for ViT-S/14).
            hidden_channels: Width of the conv stack.
            convolution_layers: Number of 3x3 conv + GELU layers (2..3 recommended).
            class_count: Number of patch classes.
        """
        super().__init__()
        layer_list: list[nn.Module] = [nn.Conv2d(feature_channels, hidden_channels, 1), nn.GELU()]
        for _ in range(convolution_layers):
            layer_list += [
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                nn.GroupNorm(8, hidden_channels),
                nn.GELU(),
            ]
        self.feature_stack = nn.Sequential(*layer_list)
        self.patch_classifier = nn.Conv2d(hidden_channels, class_count, 1)
        self.page_classifier = nn.Linear(2 * hidden_channels, 1)

    def forward(self, patch_features: torch.Tensor, valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Classify every patch and the whole page.

        Args:
            patch_features: (B, feature_channels, rows, columns) backbone grid.
            valid_mask: (B, 1, rows, columns) float, 1 inside the page, 0 in batch padding.

        Returns:
            ``(patch_logits (B, classes, rows, columns), page_logit (B,))``.
        """
        hidden_features = self.feature_stack(patch_features) * valid_mask
        patch_logits = self.patch_classifier(hidden_features)
        valid_count = valid_mask.sum(dim=(2, 3)).clamp_min(1.0)
        mean_pooled = hidden_features.sum(dim=(2, 3)) / valid_count
        max_pooled = hidden_features.masked_fill(valid_mask == 0, -1e4).amax(dim=(2, 3))
        page_logit = self.page_classifier(torch.cat([mean_pooled, max_pooled], dim=1)).squeeze(1)
        return patch_logits, page_logit
