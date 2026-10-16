"""Geometric filter that tells hand-drawn lines from printed rules (STAGE_CLEAN.md §6, post-processing).

Printed rules are straight, uniform in thickness, flat (horizontal/vertical) and end exactly at
the word or column edges. Hand lines wobble, vary in thickness, slope and overshoot the words.
"""

from dataclasses import dataclass

import numpy

from readermarks.settings.model_settings import CleaningSettings
from readermarks.synthesis.page_analysis import PixelBox

PRINTED_RULE_VERDICT = "printed_rule"
HAND_LINE_VERDICT = "hand_line"
NOT_LINE_VERDICT = "not_line"
MINIMUM_ELONGATION = 8.0


@dataclass(frozen=True)
class LineMeasurement:
    """Shape statistics of one line-like component, measured along its main axis.

    Attributes:
        length: Extent along the main axis in pixels.
        mean_thickness: Mean thickness across the axis in pixels.
        thickness_variation: Coefficient of variation of the thickness.
        fit_residual: RMS distance of the centerline from a straight fit, pixels.
        slope_degrees: Angle of the fit to the nearest page axis, degrees.
        axis_start: Start coordinate along the main axis.
        axis_end: End coordinate along the main axis.
        horizontal: True when the main axis is horizontal.
    """

    axis_length: float
    mean_thickness: float
    thickness_variation: float
    fit_residual: float
    slope_degrees: float
    axis_start: int
    axis_end: int
    horizontal: bool


def measure_line(component_mask: numpy.ndarray) -> LineMeasurement | None:
    """Measure a component as a line along its longer bounding-box side.

    Args:
        component_mask: (h, w) bool mask cropped to the component's bounding box.

    Returns:
        Measurement, or None when the component is not elongated enough to be a line.
    """
    horizontal = component_mask.shape[1] >= component_mask.shape[0]
    oriented_mask = component_mask if horizontal else component_mask.T
    column_counts = oriented_mask.sum(axis=0)
    filled_columns = numpy.nonzero(column_counts)[0]
    if filled_columns.size < 2:
        return None
    row_positions = numpy.arange(oriented_mask.shape[0])[:, None]
    centerline = (oriented_mask * row_positions).sum(axis=0)[filled_columns] / column_counts[filled_columns]
    thickness = column_counts[filled_columns].astype(numpy.float64)
    axis_length = float(filled_columns[-1] - filled_columns[0] + 1)
    if axis_length / max(1.0, thickness.mean()) < MINIMUM_ELONGATION:
        return None
    line_slope, line_intercept = numpy.polyfit(filled_columns, centerline, 1)
    fit_residual = float(numpy.sqrt(numpy.mean((centerline - (line_slope * filled_columns + line_intercept)) ** 2)))
    return LineMeasurement(
        axis_length,
        float(thickness.mean()),
        float(thickness.std() / max(1e-6, thickness.mean())),
        fit_residual,
        float(numpy.degrees(numpy.arctan(abs(line_slope)))),
        int(filled_columns[0]),
        int(filled_columns[-1]),
        horizontal,
    )


def overshoots_words(component_box: PixelBox, word_boxes: list[PixelBox], minimum_overshoot: int) -> bool:
    """Whether a horizontal line runs past the words it underlines (a hand-drawn trait).

    Args:
        component_box: Box of the line component.
        word_boxes: Printed word boxes of the page.
        minimum_overshoot: Overshoot in pixels that counts as hand-drawn.

    Returns:
        True when an end of the line sticks out past the nearest word edge above it by more
        than ``minimum_overshoot`` pixels while the other end touches a word.
    """
    words_above = [
        word_box
        for word_box in word_boxes
        if 0 <= component_box[1] - word_box[3] <= 8
        and word_box[2] > component_box[0]
        and word_box[0] < component_box[2]
    ]
    if not words_above:
        return False
    left_overshoot = min(word_box[0] for word_box in words_above) - component_box[0]
    right_overshoot = component_box[2] - max(word_box[2] for word_box in words_above)
    return max(left_overshoot, right_overshoot) > minimum_overshoot


def classify_line(
    component_mask: numpy.ndarray,
    component_box: PixelBox,
    word_boxes: list[PixelBox],
    cleaning_settings: CleaningSettings,
) -> str:
    """Decide whether a mark component is a printed rule, a hand line or not a line at all.

    Usage:
        Called by mask fusion for every mark component; ``printed_rule`` verdicts are vetoed
        (kept on the page and drawn green in the debug overlay).

    Args:
        component_mask: (h, w) bool mask cropped to the component.
        component_box: Box of the component on the page.
        word_boxes: Printed word boxes (from the printed-ink mask).
        cleaning_settings: Thresholds of the filter.

    Returns:
        ``PRINTED_RULE_VERDICT``, ``HAND_LINE_VERDICT`` or ``NOT_LINE_VERDICT``.
    """
    line_measurement = measure_line(component_mask)
    if line_measurement is None:
        return NOT_LINE_VERDICT
    looks_printed = (
        line_measurement.fit_residual <= cleaning_settings.rule_maximum_residual
        and line_measurement.thickness_variation <= cleaning_settings.rule_maximum_thickness_variation
        and line_measurement.slope_degrees <= cleaning_settings.rule_maximum_slope_degrees
        and line_measurement.axis_length >= cleaning_settings.rule_minimum_length
    )
    if looks_printed and line_measurement.horizontal:
        looks_printed = not overshoots_words(component_box, word_boxes, cleaning_settings.hand_minimum_overshoot)
    return PRINTED_RULE_VERDICT if looks_printed else HAND_LINE_VERDICT
