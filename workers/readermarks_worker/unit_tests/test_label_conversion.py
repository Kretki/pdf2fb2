"""Label Studio conversion to pixel masks and patch labels."""

import numpy

from readermarks.datasets.patch_grid import compute_geometry, masks_to_patch_labels
from readermarks.labeling.label_conversion import convert_task
from readermarks.settings.class_catalog import SPOTTER_CLASSES, spotter_channel

PAGE_WIDTH, PAGE_HEIGHT = 800, 1400


def label_result(result_type: str, label_value: dict) -> dict:
    """One Label Studio result item.

    Args:
        result_type: Result type such as ``rectanglelabels``.
        label_value: ``value`` payload.

    Returns:
        Result dict.
    """
    return {"type": result_type, "value": label_value, "original_width": PAGE_WIDTH, "original_height": PAGE_HEIGHT}


def test_rectangle_and_polyline_are_rasterised() -> None:
    """A box and a polyline land in their classes at the right place."""
    label_task = {
        "data": {"image": "page_0016.png"},
        "annotations": [
            {
                "result": [
                    label_result(
                        "rectanglelabels",
                        {"x": 10, "y": 10, "width": 5, "height": 5, "rectanglelabels": ["hand_text_margin"]},
                    ),
                    label_result(
                        "polylinelabels", {"points": [[20, 50], [60, 50.5]], "polylinelabels": ["hand_underline"]}
                    ),
                ]
            }
        ],
    }
    pixel_masks, patch_labels = convert_task(label_task, 1344, 14, 0.02)
    assert pixel_masks.shape == (len(SPOTTER_CLASSES), PAGE_HEIGHT, PAGE_WIDTH)
    margin_mask = pixel_masks[spotter_channel("hand_text_margin")]
    assert margin_mask[150, 100] == 255 and margin_mask[300, 300] == 0
    underline_mask = pixel_masks[spotter_channel("hand_underline")]
    assert underline_mask[703, 300] == 255
    assert patch_labels[spotter_channel("hand_underline")].sum() > 0
    assert patch_labels[spotter_channel("printed_rule")].sum() == 0


def test_unknown_classes_and_empty_tasks_are_ignored() -> None:
    """Unknown labels are skipped; tasks without results convert to None."""
    unknown_task = {
        "annotations": [
            {
                "result": [
                    label_result(
                        "rectanglelabels",
                        {"x": 1, "y": 1, "width": 1, "height": 1, "rectanglelabels": ["coffee_stain"]},
                    )
                ]
            }
        ]
    }
    pixel_masks, _ = convert_task(unknown_task, 1344, 14, 0.02)
    assert pixel_masks.sum() == 0
    assert convert_task({"annotations": []}, 1344, 14, 0.02) is None


def test_patch_labels_follow_coverage_fraction() -> None:
    """A patch is positive only when the covered fraction reaches the threshold."""
    grid_geometry = compute_geometry(1344, 672, 1344, 14)
    class_masks = numpy.zeros((1, 1344, 672), numpy.uint8)
    class_masks[0, 0:14, 0:14] = 255
    class_masks[0, 0:1, 14:16] = 255
    patch_labels = masks_to_patch_labels(class_masks, grid_geometry, 0.05)
    assert patch_labels[0, 0, 0] == 1 and patch_labels[0, 0, 1] == 0
    assert patch_labels.shape == (1, grid_geometry.grid_rows, grid_geometry.grid_columns)
