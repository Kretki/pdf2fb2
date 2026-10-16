"""Turn segmenter probability maps into kept mark components, protected rules and the ink mask."""

from dataclasses import dataclass

import cv2
import numpy

from readermarks.postprocess.line_geometry import HAND_LINE_VERDICT, PRINTED_RULE_VERDICT, classify_line
from readermarks.settings.class_catalog import MARK_CLASSES, segmenter_channel
from readermarks.settings.model_settings import CleaningSettings
from readermarks.synthesis.page_analysis import PixelBox, estimate_x_height, merged_boxes

RULE_OVERLAP_VETO = 0.6
PROBABILITY_SCALE = 255.0


@dataclass(frozen=True)
class MarkComponent:
    """One connected reader mark.

    Attributes:
        class_name: Most probable mark class.
        bounding_box: (x_start, y_start, x_end, y_end), end exclusive.
        confidence: Mean ``mark_any`` probability of its pixels, 0..1.
        component_mask: (h, w) bool mask cropped to the bounding box.
        line_verdict: Geometric filter verdict (``hand_line`` / ``not_line``).
    """

    class_name: str
    bounding_box: PixelBox
    confidence: float
    component_mask: numpy.ndarray
    line_verdict: str


@dataclass(frozen=True)
class FusedMarks:
    """Result of mask fusion for one page.

    Attributes:
        mark_mask: (H, W) bool, pixels of kept reader marks.
        class_masks: (7, H, W) bool, multi-label mark classes restricted to ``mark_mask``.
        ink_mask: (H, W) bool, predicted printed ink.
        protected_mask: (H, W) bool, components vetoed as printed rules.
        components: Kept mark components.
        word_boxes: Printed word boxes derived from ``ink_mask``.
    """

    mark_mask: numpy.ndarray
    class_masks: numpy.ndarray
    ink_mask: numpy.ndarray
    protected_mask: numpy.ndarray
    components: list[MarkComponent]
    word_boxes: list[PixelBox]


def channel_mask(probability_maps: numpy.ndarray, class_name: str, threshold: float) -> numpy.ndarray:
    """Threshold one channel of the uint8-scaled probability maps.

    Args:
        probability_maps: (K, H, W) uint8 probabilities scaled to 0..255.
        class_name: Segmenter class.
        threshold: Probability threshold 0..1.

    Returns:
        (H, W) bool mask.
    """
    return probability_maps[segmenter_channel(class_name)] >= threshold * PROBABILITY_SCALE


def assess_component(
    probability_maps: numpy.ndarray,
    component_mask: numpy.ndarray,
    component_box: PixelBox,
    rule_mask: numpy.ndarray,
    word_boxes: list[PixelBox],
    cleaning_settings: CleaningSettings,
) -> MarkComponent | None:
    """Classify one component, or veto it as a printed rule.

    Args:
        probability_maps: (K, H, W) uint8 probabilities.
        component_mask: (h, w) bool mask cropped to ``component_box``.
        component_box: Box of the component.
        rule_mask: (H, W) bool predicted printed rules.
        word_boxes: Printed word boxes.
        cleaning_settings: Thresholds.

    Returns:
        The component, or None when it is a printed rule.
    """
    left_edge, top_edge, right_edge, bottom_edge = component_box
    window_maps = probability_maps[:, top_edge:bottom_edge, left_edge:right_edge][:, component_mask].astype(
        numpy.float32
    )
    line_verdict = classify_line(component_mask, component_box, word_boxes, cleaning_settings)
    rule_overlap = float(rule_mask[top_edge:bottom_edge, left_edge:right_edge][component_mask].mean())
    overlaps_rule = rule_overlap >= RULE_OVERLAP_VETO and line_verdict != HAND_LINE_VERDICT
    if line_verdict == PRINTED_RULE_VERDICT or overlaps_rule:
        return None
    mark_scores = [window_maps[segmenter_channel(class_name)].mean() for class_name in MARK_CLASSES]
    confidence = float(window_maps[segmenter_channel("mark_any")].mean() / PROBABILITY_SCALE)
    return MarkComponent(
        MARK_CLASSES[int(numpy.argmax(mark_scores))], component_box, confidence, component_mask, line_verdict
    )


def printed_word_boxes(ink_mask: numpy.ndarray) -> list[PixelBox]:
    """Word boxes from the predicted printed-ink mask (no OCR).

    Args:
        ink_mask: (H, W) bool.

    Returns:
        Word boxes.
    """
    ink_bytes = ink_mask.astype(numpy.uint8) * 255
    x_height = estimate_x_height(ink_bytes)
    return merged_boxes(ink_bytes, max(3, int(0.45 * x_height)), 3, x_height)


def fuse_masks(probability_maps: numpy.ndarray, cleaning_settings: CleaningSettings) -> FusedMarks:
    """Threshold, split into components, classify and veto printed rules.

    Usage:
        Called by the page cleaner after the segmenter ran on the flagged regions.

    Args:
        probability_maps: (K, H, W) uint8 segmenter probabilities scaled to 0..255 (zero outside regions).
        cleaning_settings: Thresholds.

    Returns:
        Fused marks of the page.
    """
    raw_mark_mask = channel_mask(probability_maps, "mark_any", cleaning_settings.pixel_threshold)
    ink_mask = channel_mask(probability_maps, "printed_ink", cleaning_settings.ink_threshold)
    rule_mask = channel_mask(probability_maps, "printed_rule", 0.5)
    word_boxes = printed_word_boxes(ink_mask)
    component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(
        raw_mark_mask.astype(numpy.uint8), connectivity=8
    )
    kept_mask, protected_mask = numpy.zeros_like(raw_mark_mask), numpy.zeros_like(raw_mark_mask)
    components = []
    for component_index in range(1, component_count):
        corner_x, corner_y, box_width, box_height, pixel_count = component_stats[component_index]
        if pixel_count < cleaning_settings.minimum_component_pixels:
            continue
        component_box = (int(corner_x), int(corner_y), int(corner_x + box_width), int(corner_y + box_height))
        component_mask = (
            component_labels[corner_y : corner_y + box_height, corner_x : corner_x + box_width] == component_index
        )
        mark_component = assess_component(
            probability_maps, component_mask, component_box, rule_mask, word_boxes, cleaning_settings
        )
        target_mask = protected_mask if mark_component is None else kept_mask
        target_mask[corner_y : corner_y + box_height, corner_x : corner_x + box_width] |= component_mask
        if mark_component is not None:
            components.append(mark_component)
    class_masks = numpy.stack(
        [
            channel_mask(probability_maps, class_name, cleaning_settings.pixel_threshold) & kept_mask
            for class_name in MARK_CLASSES
        ]
    )
    return FusedMarks(kept_mask, class_masks, ink_mask, protected_mask, components, word_boxes)
