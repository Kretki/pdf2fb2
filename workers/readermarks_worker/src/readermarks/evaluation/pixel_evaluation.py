"""Pixel-level evaluation of the full pipeline on the synthetic test split (masks are exact there)."""

from pathlib import Path

import numpy

from readermarks.datasets.image_files import read_rgb_image
from readermarks.datasets.mask_archive import read_mask_archive
from readermarks.evaluation.classification_metrics import DetectionCounts
from readermarks.inference.page_cleaner import PageCleaner
from readermarks.settings.class_catalog import MARK_CLASS_COUNT, spotter_channel
from readermarks.synthesis.dataset_writer import read_manifest


def region_mask(page_shape: tuple[int, int], region_boxes: list) -> numpy.ndarray:
    """Boolean mask of the flagged regions.

    Args:
        page_shape: (height, width).
        region_boxes: Flagged regions.

    Returns:
        (H, W) bool.
    """
    covered_mask = numpy.zeros(page_shape, bool)
    for left_edge, top_edge, right_edge, bottom_edge in region_boxes:
        covered_mask[top_edge:bottom_edge, left_edge:right_edge] = True
    return covered_mask


def evaluate_synthetic_pixels(page_cleaner: PageCleaner, split_directory: Path, sample_limit: int) -> dict:
    """``mark_any`` / ``printed_ink`` pixel IoU and F1, rule safety and page accuracy on synthetic pages.

    Usage:
        Called by the evaluation service on the synthetic test split.

    Args:
        page_cleaner: Loaded cleaner.
        split_directory: Generated synthetic split.
        sample_limit: Maximum number of samples to evaluate.

    Returns:
        ``{"mark_any", "printed_ink_in_regions", "printed_rule_pixels_removed_rate", "page_level"}``.
    """
    mark_counts, ink_counts, page_counts = DetectionCounts(), DetectionCounts(), DetectionCounts()
    rule_pixels, removed_rule_pixels = 0, 0
    for manifest_record in read_manifest(split_directory)[:sample_limit]:
        rgb_image = read_rgb_image(split_directory / f"{manifest_record['sample']}.png")
        class_masks = read_mask_archive(split_directory / f"{manifest_record['sample']}_masks.npz") > 0
        page_verdict = page_cleaner.inspect_page(rgb_image)
        expected_marks = class_masks[:MARK_CLASS_COUNT].any(axis=0)
        predicted_marks = (
            page_verdict.fused_marks.mark_mask if page_verdict.has_marks else numpy.zeros_like(expected_marks)
        )
        mark_counts.add_arrays(predicted_marks, expected_marks)
        page_counts.add_arrays(numpy.array([page_verdict.has_marks]), numpy.array([bool(expected_marks.any())]))
        if page_verdict.fused_marks is not None:
            inside_regions = region_mask(expected_marks.shape, page_verdict.region_boxes)
            ink_counts.add_arrays(
                page_verdict.fused_marks.ink_mask[inside_regions],
                class_masks[spotter_channel("printed_ink")][inside_regions],
            )
        rule_only = class_masks[spotter_channel("printed_rule")] & ~expected_marks
        rule_pixels += int(rule_only.sum())
        removed_rule_pixels += int((rule_only & predicted_marks).sum())
    return {
        "mark_any": mark_counts.summarize_counts(),
        "printed_ink_in_regions": ink_counts.summarize_counts(),
        "printed_rule_pixels_removed_rate": round(removed_rule_pixels / rule_pixels, 5) if rule_pixels else None,
        "page_level": page_counts.summarize_counts(),
    }
