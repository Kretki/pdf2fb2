"""Grid-search the page/patch/pixel thresholds on the real validation pages (never on the test split).

Stage 1 runs once per page; Stage 2 runs once per patch threshold on exactly the regions that
threshold produces (as at inference), and every pixel threshold reuses those probabilities.
"""

import itertools
from dataclasses import dataclass

import numpy

from readermarks.datasets.layout_regions import read_masked_page
from readermarks.datasets.page_inventory import RealPage
from readermarks.inference.page_cleaner import PageCleaner
from readermarks.inference.segmenter_inference import SegmenterRunner
from readermarks.inference.spotter_inference import SpotterRunner, flagged_regions
from readermarks.postprocess.mask_fusion import fuse_masks
from readermarks.settings.model_settings import CleaningSettings

PAGE_THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
PATCH_THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
PIXEL_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
MINIMUM_PIXEL_COUNTS = (40, 120, 300, 600, 1200, 2500)


@dataclass(frozen=True)
class PageEvidence:
    """Threshold-independent evidence of one validation page.

    Attributes:
        expected: Ground-truth page label.
        page_probability: Spotter page probability.
        mark_pixels: Mapping (patch_threshold, pixel_threshold) -> kept mark pixels (0 without regions).
    """

    expected: bool
    page_probability: float
    mark_pixels: dict[tuple[float, float], int]


def count_mark_pixels(
    probability_maps: numpy.ndarray | None, cleaning_settings: CleaningSettings, pixel_threshold: float
) -> int:
    """Kept mark pixels after fusion at one pixel threshold.

    Args:
        probability_maps: Segmenter output, or None when no region was flagged.
        cleaning_settings: Base settings (geometric filter, sizes).
        pixel_threshold: Pixel probability threshold.

    Returns:
        Number of kept mark pixels (0 without regions).
    """
    if probability_maps is None:
        return 0
    fused_marks = fuse_masks(
        probability_maps, cleaning_settings.model_copy(update={"pixel_threshold": pixel_threshold})
    )
    return int(fused_marks.mark_mask.sum())


def collect_evidence(
    spotter_runner: SpotterRunner,
    segmenter_runner: SegmenterRunner,
    cleaning_settings: CleaningSettings,
    real_page: RealPage,
    picture_regions: dict[int, list[list[float]]],
) -> PageEvidence:
    """Run both stages and measure kept mark pixels for every patch/pixel threshold pair.

    Args:
        spotter_runner: Stage 1 runner.
        segmenter_runner: Stage 2 runner.
        cleaning_settings: Base settings (geometric filter, sizes).
        real_page: Labeled validation page.
        picture_regions: Picture boxes per page, blanked before inspection.

    Returns:
        Page evidence.
    """
    rgb_image = read_masked_page(real_page, picture_regions)
    spotter_output = spotter_runner.spot_page(rgb_image)
    ungated_settings = cleaning_settings.model_copy(update={"page_threshold": 0.0})
    regions_by_patch = {
        patch_threshold: flagged_regions(
            spotter_output, ungated_settings.model_copy(update={"patch_threshold": patch_threshold})
        )
        for patch_threshold in PATCH_THRESHOLDS
    }
    mark_pixels = {}
    for patch_threshold, region_boxes in regions_by_patch.items():
        probability_maps = segmenter_runner.segment_regions(rgb_image, region_boxes) if region_boxes else None
        for pixel_threshold in PIXEL_THRESHOLDS:
            mark_pixels[(patch_threshold, pixel_threshold)] = count_mark_pixels(
                probability_maps, cleaning_settings, pixel_threshold
            )
    return PageEvidence(bool(real_page.has_marks), spotter_output.page_probability, mark_pixels)


def score_combination(page_evidence: list[PageEvidence], threshold_values: tuple[float, float, float, int]) -> tuple:
    """Sort key of a threshold combination: accuracy, then fewer false positives, then central values.

    Args:
        page_evidence: Evidence of all validation pages.
        threshold_values: (page, patch, pixel, minimum pixels).

    Returns:
        Tuple to maximise.
    """
    page_threshold, patch_threshold, pixel_threshold, minimum_pixels = threshold_values
    predictions = [
        evidence.page_probability >= page_threshold
        and evidence.mark_pixels[(patch_threshold, pixel_threshold)] >= minimum_pixels
        for evidence in page_evidence
    ]
    correct_count = sum(
        predicted == evidence.expected for predicted, evidence in zip(predictions, page_evidence, strict=True)
    )
    false_positives = sum(
        predicted and not evidence.expected for predicted, evidence in zip(predictions, page_evidence, strict=True)
    )
    centrality = -abs(page_threshold - 0.5) - abs(patch_threshold - 0.5) - abs(pixel_threshold - 0.5)
    return correct_count, -false_positives, centrality


def tune_thresholds(
    page_cleaner: PageCleaner,
    spotter_runner: SpotterRunner,
    segmenter_runner: SegmenterRunner,
    validation_pages: list[RealPage],
    picture_regions: dict[int, list[list[float]]],
) -> dict:
    """Pick the threshold combination with the best page-level accuracy on validation pages.

    Usage:
        ``python -m readermarks tune`` (writes ``thresholds.json``).

    Args:
        page_cleaner: Cleaner (provides the base cleaning settings).
        spotter_runner: Stage 1 runner.
        segmenter_runner: Stage 2 runner.
        validation_pages: Labeled ``real_validation`` pages.
        picture_regions: Picture boxes per page, blanked before inspection.

    Returns:
        ``{"thresholds": {...}, "validation_accuracy": float, "page_count": int}``.
    """
    page_evidence = [
        collect_evidence(spotter_runner, segmenter_runner, page_cleaner.cleaning_settings, real_page, picture_regions)
        for real_page in validation_pages
    ]
    best_values = max(
        itertools.product(PAGE_THRESHOLDS, PATCH_THRESHOLDS, PIXEL_THRESHOLDS, MINIMUM_PIXEL_COUNTS),
        key=lambda threshold_values: score_combination(page_evidence, threshold_values),
    )
    best_correct = score_combination(page_evidence, best_values)[0]
    return {
        "thresholds": {
            "page_threshold": best_values[0],
            "patch_threshold": best_values[1],
            "pixel_threshold": best_values[2],
            "minimum_mark_pixels": best_values[3],
        },
        "validation_accuracy": round(best_correct / max(1, len(page_evidence)), 4),
        "page_count": len(page_evidence),
        "page_probabilities": [round(float(numpy.float32(evidence.page_probability)), 4) for evidence in page_evidence],
    }
