"""Convert Label Studio box/polygon/polyline labels into pixel masks and Stage 1 patch labels.

Supported result types: ``rectanglelabels``, ``polygonlabels`` and ``polylinelabels``; coordinates are
percentages of the image size (Label Studio convention). Polylines are buffered by the estimated
stroke width; boxes of line-like classes are filled (coarse masks for validation).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy

from readermarks.datasets.patch_grid import compute_geometry, masks_to_patch_labels
from readermarks.settings.class_catalog import SPOTTER_CLASSES, spotter_channel

MAX_EXPORT_BYTES = 64 * 1024 * 1024
DEFAULT_STROKE_WIDTH = 3
SUPPORTED_TYPES = frozenset({"rectanglelabels", "polygonlabels", "polylinelabels"})


@dataclass(frozen=True)
class LabeledShape:
    """One labeled region in page pixels.

    Attributes:
        class_name: One of ``SPOTTER_CLASSES``.
        shape_kind: ``rectangle``, ``polygon`` or ``polyline``.
        points: (N, 2) float32 pixel coordinates (rectangles: 4 corners).
    """

    class_name: str
    shape_kind: str
    shape_points: numpy.ndarray


def result_to_shape(label_result: dict, image_width: int, image_height: int) -> LabeledShape | None:
    """Decode one Label Studio result item.

    Args:
        label_result: Item of ``annotations[].result``.
        image_width: Page width in pixels.
        image_height: Page height in pixels.

    Returns:
        The shape, or None for unsupported types or unknown classes.
    """
    result_type = label_result.get("type")
    label_value = label_result.get("value", {})
    if result_type not in SUPPORTED_TYPES:
        return None
    class_names = label_value.get(result_type, [])
    if not class_names or class_names[0] not in SPOTTER_CLASSES:
        return None
    percent_scale = numpy.array([image_width / 100.0, image_height / 100.0], numpy.float32)
    if result_type == "rectanglelabels":
        left_edge, top_edge = label_value["x"], label_value["y"]
        right_edge, bottom_edge = left_edge + label_value["width"], top_edge + label_value["height"]
        corner_points = numpy.array(
            [[left_edge, top_edge], [right_edge, top_edge], [right_edge, bottom_edge], [left_edge, bottom_edge]],
            numpy.float32,
        )
        return LabeledShape(class_names[0], "rectangle", corner_points * percent_scale)
    shape_kind = "polygon" if result_type == "polygonlabels" else "polyline"
    return LabeledShape(class_names[0], shape_kind, numpy.array(label_value["points"], numpy.float32) * percent_scale)


def rasterise_shapes(
    labeled_shapes: list[LabeledShape], image_height: int, image_width: int, stroke_width: int
) -> numpy.ndarray:
    """Draw shapes into per-class masks.

    Args:
        labeled_shapes: Shapes of one page.
        image_height: Page height.
        image_width: Page width.
        stroke_width: Buffer width of polylines in pixels.

    Returns:
        (len(SPOTTER_CLASSES), H, W) uint8 masks, 255 = labeled.
    """
    class_masks = numpy.zeros((len(SPOTTER_CLASSES), image_height, image_width), numpy.uint8)
    for labeled_shape in labeled_shapes:
        class_mask = class_masks[spotter_channel(labeled_shape.class_name)]
        integer_points = numpy.round(labeled_shape.shape_points).astype(numpy.int32).reshape(-1, 1, 2)
        if labeled_shape.shape_kind == "polyline":
            cv2.polylines(class_mask, [integer_points], False, 255, max(1, stroke_width))
        else:
            cv2.fillPoly(class_mask, [integer_points], 255)
    return class_masks


def read_label_studio_export(export_path: Path) -> list[dict]:
    """Read a Label Studio JSON export (list of tasks), bounded in size.

    Args:
        export_path: JSON export file.

    Returns:
        Task dicts.

    Raises:
        ValueError: If the file is too large or not a list.
    """
    if export_path.stat().st_size > MAX_EXPORT_BYTES:
        raise ValueError(f"{export_path} is larger than {MAX_EXPORT_BYTES} bytes.")
    task_list = json.loads(export_path.read_text(encoding="utf-8"))
    if not isinstance(task_list, list):
        raise ValueError(f"{export_path} must contain a JSON list of tasks.")
    return task_list


def task_shapes(label_task: dict) -> tuple[list[LabeledShape], int, int]:
    """Shapes of the first annotation of a task, with the original image size.

    Args:
        label_task: Task dict from the export.

    Returns:
        ``(shapes, image_width, image_height)``; size is (0, 0) when the task has no results.
    """
    annotations = label_task.get("annotations") or []
    label_results = annotations[0].get("result", []) if annotations else []
    if not label_results:
        return [], 0, 0
    image_width = int(label_results[0]["original_width"])
    image_height = int(label_results[0]["original_height"])
    shape_list = [result_to_shape(label_result, image_width, image_height) for label_result in label_results]
    return [labeled_shape for labeled_shape in shape_list if labeled_shape is not None], image_width, image_height


def convert_task(
    label_task: dict, long_side: int, patch_size: int, label_fraction: float, stroke_width: int = DEFAULT_STROKE_WIDTH
) -> tuple[numpy.ndarray, numpy.ndarray] | None:
    """Pixel masks and patch labels of one labeled task.

    Usage:
        Called by the ``convert-labels`` command for every task of an export.

    Args:
        label_task: Task dict.
        long_side: Spotter input long side.
        patch_size: Spotter patch size.
        label_fraction: Patch-positive coverage fraction.
        stroke_width: Polyline buffer width in pixels.

    Returns:
        ``(pixel_masks (C, H, W) uint8, patch_labels (C, rows, columns) uint8)``, or None without labels.
    """
    labeled_shapes, image_width, image_height = task_shapes(label_task)
    if not image_width:
        return None
    pixel_masks = rasterise_shapes(labeled_shapes, image_height, image_width, stroke_width)
    grid_geometry = compute_geometry(image_height, image_width, long_side, patch_size)
    return pixel_masks, masks_to_patch_labels(pixel_masks, grid_geometry, label_fraction)
