"""Build the JSON records of ``marks.json`` from fused mark components."""

import cv2
import numpy

from readermarks.postprocess.line_geometry import HAND_LINE_VERDICT
from readermarks.postprocess.mask_fusion import FusedMarks, MarkComponent
from readermarks.synthesis.page_analysis import PixelBox

TOUCH_MARGIN = 3
CONTOUR_TOLERANCE = 1.5
POLYLINE_SAMPLES = 16


def component_contour(mark_component: MarkComponent) -> list[list[int]]:
    """Simplified outer contour of a component in page coordinates.

    Args:
        mark_component: Kept mark.

    Returns:
        List of ``[x, y]`` points of the largest outer contour.
    """
    contours, _ = cv2.findContours(
        mark_component.component_mask.astype(numpy.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    largest_contour = cv2.approxPolyDP(max(contours, key=cv2.contourArea), CONTOUR_TOLERANCE, True)
    left_edge, top_edge = mark_component.bounding_box[:2]
    return [
        [int(contour_point[0][0]) + left_edge, int(contour_point[0][1]) + top_edge] for contour_point in largest_contour
    ]


def component_polyline(mark_component: MarkComponent) -> list[list[float]]:
    """Centerline of a line-like component sampled along its longer side.

    Args:
        mark_component: Kept mark.

    Returns:
        ``[x, y]`` points (empty for components that are not lines).
    """
    if mark_component.line_verdict != HAND_LINE_VERDICT:
        return []
    component_mask = mark_component.component_mask
    horizontal = component_mask.shape[1] >= component_mask.shape[0]
    oriented_mask = component_mask if horizontal else component_mask.T
    filled_columns = numpy.nonzero(oriented_mask.any(axis=0))[0]
    sampled_columns = filled_columns[numpy.linspace(0, filled_columns.size - 1, POLYLINE_SAMPLES).astype(int)]
    left_edge, top_edge = mark_component.bounding_box[:2]
    polyline_points = []
    for column_index in sampled_columns:
        center_row = float(numpy.nonzero(oriented_mask[:, column_index])[0].mean())
        polyline_x, polyline_y = (column_index, center_row) if horizontal else (center_row, column_index)
        polyline_points.append([round(float(polyline_x) + left_edge, 1), round(float(polyline_y) + top_edge, 1)])
    return polyline_points


def touched_words(mark_component: MarkComponent, word_boxes: list[PixelBox]) -> list[list[int]]:
    """Printed word boxes that the mark touches or lies directly against.

    Args:
        mark_component: Kept mark.
        word_boxes: Word boxes from the printed-ink mask.

    Returns:
        List of ``[x_start, y_start, x_end, y_end]`` word boxes.
    """
    left_edge, top_edge, right_edge, bottom_edge = mark_component.bounding_box
    dilated_mask = (
        cv2.dilate(
            mark_component.component_mask.astype(numpy.uint8), numpy.ones((2 * TOUCH_MARGIN + 1,) * 2, numpy.uint8)
        )
        > 0
    )
    touched_boxes = []
    for word_box in word_boxes:
        overlap_x0, overlap_y0 = max(word_box[0], left_edge - TOUCH_MARGIN), max(word_box[1], top_edge - TOUCH_MARGIN)
        overlap_x1, overlap_y1 = (
            min(word_box[2], right_edge + TOUCH_MARGIN),
            min(word_box[3], bottom_edge + TOUCH_MARGIN),
        )
        if overlap_x1 <= overlap_x0 or overlap_y1 <= overlap_y0:
            continue
        local_rows = slice(max(0, overlap_y0 - top_edge), max(0, min(bottom_edge - top_edge, overlap_y1 - top_edge)))
        local_columns = slice(
            max(0, overlap_x0 - left_edge), max(0, min(right_edge - left_edge, overlap_x1 - left_edge))
        )
        if dilated_mask[local_rows, local_columns].any():
            touched_boxes.append([int(coordinate) for coordinate in word_box])
    return touched_boxes


def build_mark_records(fused_marks: FusedMarks) -> list[dict]:
    """JSON-ready records of all kept marks.

    Usage:
        Called by the page cleaner to write ``marks.json``.

    Args:
        fused_marks: Fusion result of the page.

    Returns:
        One dict per mark: class, bbox, confidence, contour, polyline, touched_words.
    """
    return [
        {
            "class": mark_component.class_name,
            "bbox": list(mark_component.bounding_box),
            "confidence": round(mark_component.confidence, 4),
            "contour": component_contour(mark_component),
            "polyline": component_polyline(mark_component),
            "touched_words": touched_words(mark_component, fused_marks.word_boxes),
        }
        for mark_component in fused_marks.components
    ]
