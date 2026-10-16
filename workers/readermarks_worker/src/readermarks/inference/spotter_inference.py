"""Stage 1 inference: patch probabilities, page probability and flagged pixel regions."""

from dataclasses import dataclass

import cv2
import numpy
import torch

from readermarks.datasets.patch_grid import GridGeometry, compute_geometry, patch_box_to_pixels, prepare_spotter_image
from readermarks.networks.backbone_wrapper import PatchBackbone, normalise_image
from readermarks.networks.spotter_head import SpotterHead
from readermarks.settings.class_catalog import MARK_CLASS_COUNT
from readermarks.settings.model_settings import CleaningSettings, SpotterSettings
from readermarks.synthesis.page_analysis import PixelBox


@dataclass(frozen=True)
class SpotterOutput:
    """Stage 1 result for one page.

    Attributes:
        patch_probabilities: (classes, rows, columns) float32 sigmoid outputs.
        page_probability: Page-level ``has_marks`` probability, 0..1.
        grid_geometry: Mapping between patches and page pixels.
    """

    patch_probabilities: numpy.ndarray
    page_probability: float
    grid_geometry: GridGeometry

    @property
    def mark_probability(self) -> numpy.ndarray:
        """Return the per-patch probability of any reader mark (max over mark classes).

        Returns:
            (rows, columns) float32.
        """
        return self.patch_probabilities[:MARK_CLASS_COUNT].max(axis=0)


class SpotterRunner:
    """Runs the frozen backbone and the trained head on full pages.

    Usage:
        Built by the model loader; ``spot_page(rgb_image)`` per page.
    """

    def __init__(
        self,
        backbone: PatchBackbone,
        spotter_head: SpotterHead,
        spotter_settings: SpotterSettings,
        compute_device: torch.device,
    ) -> None:
        """Store the models.

        Args:
            backbone: Frozen patch backbone.
            spotter_head: Trained head (moved to ``compute_device``, eval mode).
            spotter_settings: Input size and patch size.
            compute_device: Device.
        """
        self._backbone = backbone
        self._spotter_head = spotter_head.eval().to(compute_device)
        self._spotter_settings = spotter_settings
        self._compute_device = compute_device

    @torch.inference_mode()
    def spot_features(self, patch_features: torch.Tensor, grid_geometry: GridGeometry) -> SpotterOutput:
        """Run the head on already computed (e.g. cached) features.

        Args:
            patch_features: (1, D, rows, columns) features.
            grid_geometry: Geometry of the page.

        Returns:
            Stage 1 output.
        """
        patch_features = patch_features.to(self._compute_device).float()
        valid_mask = torch.ones(1, 1, *patch_features.shape[2:], device=self._compute_device)
        patch_logits, page_logit = self._spotter_head(patch_features, valid_mask)
        return SpotterOutput(
            torch.sigmoid(patch_logits)[0].float().cpu().numpy(), float(torch.sigmoid(page_logit)[0]), grid_geometry
        )

    def spot_page(self, rgb_image: numpy.ndarray) -> SpotterOutput:
        """Run Stage 1 on a page.

        Args:
            rgb_image: (H, W, 3) uint8 page.

        Returns:
            Stage 1 output.
        """
        grid_geometry = compute_geometry(
            rgb_image.shape[0],
            rgb_image.shape[1],
            self._spotter_settings.input_long_side,
            self._spotter_settings.patch_size,
        )
        image_tensor = normalise_image(prepare_spotter_image(rgb_image, grid_geometry))
        return self.spot_features(self._backbone.extract_patch_grid(image_tensor), grid_geometry)


def flagged_regions(spotter_output: SpotterOutput, cleaning_settings: CleaningSettings) -> list[PixelBox]:
    """Connected components of above-threshold mark patches, dilated, as native pixel boxes.

    Usage:
        Called by the page cleaner; an empty list means the page passes through unchanged.

    Args:
        spotter_output: Stage 1 output.
        cleaning_settings: Patch threshold, minimum size and dilation.

    Returns:
        Pixel boxes (x_start, y_start, x_end, y_end) of the regions to segment.
    """
    if spotter_output.page_probability < cleaning_settings.page_threshold:
        return []
    flagged_grid = (spotter_output.mark_probability >= cleaning_settings.patch_threshold).astype(numpy.uint8)
    component_count, _, component_stats, _ = cv2.connectedComponentsWithStats(flagged_grid, connectivity=8)
    dilation = cleaning_settings.region_dilation_patches
    grid_rows, grid_columns = flagged_grid.shape
    region_boxes = []
    for column_start, row_start, box_width, box_height, patch_count in component_stats[1:component_count]:
        if patch_count < cleaning_settings.minimum_flagged_patches:
            continue
        region_boxes.append(
            patch_box_to_pixels(
                max(0, row_start - dilation),
                min(grid_rows, row_start + box_height + dilation),
                max(0, column_start - dilation),
                min(grid_columns, column_start + box_width + dilation),
                spotter_output.grid_geometry,
            )
        )
    return region_boxes
