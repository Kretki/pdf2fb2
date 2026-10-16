"""Patch-level metrics of the spotter on cached synthetic features (comparable to Pandey & Harit)."""

from pathlib import Path

import numpy
import torch

from readermarks.datasets.spotter_dataset import CachedSpotterDataset
from readermarks.evaluation.classification_metrics import DetectionCounts
from readermarks.inference.spotter_inference import SpotterRunner
from readermarks.settings.class_catalog import MARK_CLASS_COUNT, SPOTTER_CLASSES, spotter_channel

PAPER_BASELINE = {
    "hand_underline": {"f_score": 0.73, "precision": 0.70},
    "mark_any": {"f_score": 0.81},
    "margin_mark": {"f_score": 0.77},
    "hand_encirclement": {"f_score": 0.75},
}


def evaluate_patches(spotter_runner: SpotterRunner, cache_directory: Path, patch_threshold: float) -> dict:
    """Per-class patch precision/recall/F plus ``mark_any`` and the printed-rule false-flag rate.

    Usage:
        Called by the evaluation service on the synthetic test split.

    Args:
        spotter_runner: Runner whose head is evaluated (backbone not used: features are cached).
        cache_directory: Cached features + labels of the split.
        patch_threshold: Probability threshold for a positive patch.

    Returns:
        ``{"classes": {name: metrics}, "mark_any": metrics, "printed_rule_flagged_rate": float | None}``.
    """
    evaluation_data = CachedSpotterDataset(cache_directory)
    class_counts = {class_name: DetectionCounts() for class_name in SPOTTER_CLASSES}
    mark_counts = DetectionCounts()
    rule_patches, flagged_rule_patches = 0, 0
    for sample_index in range(len(evaluation_data)):
        patch_features, patch_labels, _ = evaluation_data[sample_index]
        spotter_output = spotter_runner.spot_features(patch_features.unsqueeze(0), None)
        predicted = spotter_output.patch_probabilities >= patch_threshold
        expected = patch_labels.numpy() > 0.5
        for class_index, class_name in enumerate(SPOTTER_CLASSES):
            class_counts[class_name].add_arrays(predicted[class_index], expected[class_index])
        predicted_mark = spotter_output.mark_probability >= patch_threshold
        expected_mark = expected[:MARK_CLASS_COUNT].any(axis=0)
        mark_counts.add_arrays(predicted_mark, expected_mark)
        rule_only = expected[spotter_channel("printed_rule")] & ~expected_mark
        rule_patches += int(rule_only.sum())
        flagged_rule_patches += int((rule_only & predicted_mark).sum())
    return {
        "classes": {
            class_name: detection_counts.summarize_counts() for class_name, detection_counts in class_counts.items()
        },
        "mark_any": mark_counts.summarize_counts(),
        "printed_rule_flagged_rate": round(flagged_rule_patches / rule_patches, 5) if rule_patches else None,
        "paper_baseline": PAPER_BASELINE,
    }


def free_cuda_cache() -> None:
    """Release cached CUDA memory between evaluation phases.

    Returns:
        None.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def mean_or_none(number_list: list[float]) -> float | None:
    """Mean rounded to 4 digits, or None for an empty list.

    Args:
        number_list: Numbers.

    Returns:
        Mean or None.
    """
    return round(float(numpy.mean(number_list)), 4) if number_list else None
