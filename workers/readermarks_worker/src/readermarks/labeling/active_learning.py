"""Active learning: rank unlabeled pages by model uncertainty and export pre-filled Label Studio tasks."""

import json
from pathlib import Path

import numpy

from readermarks.datasets.layout_regions import read_masked_page
from readermarks.datasets.page_inventory import RealPage
from readermarks.inference.page_cleaner import PageCleaner, PageVerdict

UNCERTAIN_BAND = 0.25


def page_uncertainty(page_verdict: PageVerdict, patch_threshold: float) -> float:
    """Uncertainty of a page: closeness of the page probability to 0.5 plus borderline patches.

    Args:
        page_verdict: Cleaner verdict.
        patch_threshold: Patch threshold.

    Returns:
        Score; higher = more uncertain.
    """
    page_term = 1.0 - 2.0 * abs(page_verdict.spotter_output.page_probability - 0.5)
    mark_probability = page_verdict.spotter_output.mark_probability
    borderline = numpy.abs(mark_probability - patch_threshold) < UNCERTAIN_BAND * patch_threshold
    disagreement = float(page_verdict.region_boxes != [] and not page_verdict.has_marks)
    return page_term + 0.02 * float(borderline.sum()) + disagreement


def prediction_results(page_verdict: PageVerdict, image_width: int, image_height: int) -> list[dict]:
    """Model marks as Label Studio rectangle results (percent coordinates).

    Args:
        page_verdict: Cleaner verdict.
        image_width: Page width.
        image_height: Page height.

    Returns:
        Result dicts.
    """
    if not page_verdict.has_marks:
        return []
    label_results = []
    for component_index, mark_component in enumerate(page_verdict.fused_marks.components):
        left_edge, top_edge, right_edge, bottom_edge = mark_component.bounding_box
        label_results.append(
            {
                "id": f"mark{component_index}",
                "type": "rectanglelabels",
                "from_name": "label",
                "to_name": "image",
                "original_width": image_width,
                "original_height": image_height,
                "value": {
                    "x": 100.0 * left_edge / image_width,
                    "y": 100.0 * top_edge / image_height,
                    "width": 100.0 * (right_edge - left_edge) / image_width,
                    "height": 100.0 * (bottom_edge - top_edge) / image_height,
                    "rectanglelabels": [mark_component.class_name],
                },
                "score": mark_component.confidence,
            }
        )
    return label_results


def export_uncertain_pages(
    page_cleaner: PageCleaner,
    candidate_pages: list[RealPage],
    task_count: int,
    export_path: Path,
    picture_regions: dict[int, list[list[float]]],
) -> list[dict]:
    """Run the model on candidate pages and export the most uncertain ones as Label Studio tasks.

    Usage:
        ``python -m readermarks active-learning`` (default pool: the unlabeled real pages).

    Args:
        page_cleaner: Loaded cleaner.
        candidate_pages: Unlabeled pages.
        task_count: Number of pages to export (e.g. 30).
        export_path: JSON file of the Label Studio tasks (created).
        picture_regions: Picture boxes per page, blanked before inspection.

    Returns:
        Ranking rows ``{"image", "uncertainty", "page_probability", "has_marks"}`` of the exported pages.
    """
    ranked_rows, label_tasks = [], {}
    for candidate_page in candidate_pages:
        candidate_path = candidate_page.image_path
        rgb_image = read_masked_page(candidate_page, picture_regions)
        page_verdict = page_cleaner.inspect_page(rgb_image)
        uncertainty = page_uncertainty(page_verdict, page_cleaner.cleaning_settings.patch_threshold)
        ranked_rows.append(
            {
                "image": str(candidate_path),
                "uncertainty": round(uncertainty, 4),
                "page_probability": round(page_verdict.spotter_output.page_probability, 4),
                "has_marks": page_verdict.has_marks,
            }
        )
        label_tasks[str(candidate_path)] = {
            "data": {"image": str(candidate_path)},
            "predictions": [
                {
                    "model_version": "readermarks",
                    "score": round(uncertainty, 4),
                    "result": prediction_results(page_verdict, rgb_image.shape[1], rgb_image.shape[0]),
                }
            ],
        }
    ranked_rows.sort(key=lambda ranked_row: ranked_row["uncertainty"], reverse=True)
    selected_rows = ranked_rows[:task_count]
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_path.write_text(
        json.dumps([label_tasks[ranked_row["image"]] for ranked_row in selected_rows], indent=1), encoding="utf-8"
    )
    return selected_rows
