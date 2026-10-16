"""High-level labeling actions used by the CLI: convert Label Studio exports, active-learning export."""

from pathlib import Path, PurePosixPath

import numpy

from readermarks.datasets.layout_regions import picture_regions_of
from readermarks.datasets.page_inventory import load_or_create_split, load_real_pages, pages_of_split
from readermarks.inference.model_loading import load_models
from readermarks.labeling.active_learning import export_uncertain_pages
from readermarks.labeling.label_conversion import convert_task, read_label_studio_export
from readermarks.settings.class_catalog import SPOTTER_CLASSES
from readermarks.settings.model_settings import ReaderMarkSettings
from readermarks.training.training_service import compute_device

ACTIVE_LEARNING_DIRECTORY = "active_learning"


def task_stem(label_task: dict, task_index: int) -> str:
    """Safe output file stem of a task (image file stem, never a path).

    Args:
        label_task: Task dict.
        task_index: Position in the export.

    Returns:
        File stem made of the image name's stem, or ``task_<index>``.
    """
    image_reference = str(label_task.get("data", {}).get("image", ""))
    image_stem = PurePosixPath(image_reference.replace("\\", "/")).stem
    safe_stem = "".join(character for character in image_stem if character.isalnum() or character in "_-")
    return safe_stem or f"task_{task_index}"


def convert_export(settings: ReaderMarkSettings, export_path: Path, output_directory: Path) -> dict:
    """Convert every labeled task of an export to ``<stem>_labels.npz`` (pixel masks + patch labels).

    Usage:
        ``python -m readermarks convert-labels --export labels.json --out out_dir/``.

    Args:
        settings: Pipeline settings.
        export_path: Label Studio JSON export.
        output_directory: Output directory (created).

    Returns:
        ``{"tasks": n, "converted": m, "class_pixels": {...}}``.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    label_tasks = read_label_studio_export(export_path)
    class_pixels = dict.fromkeys(SPOTTER_CLASSES, 0)
    converted_count = 0
    for task_index, label_task in enumerate(label_tasks):
        conversion = convert_task(
            label_task,
            settings.spotter_settings.input_long_side,
            settings.spotter_settings.patch_size,
            settings.spotter_settings.patch_label_fraction,
        )
        if conversion is None:
            continue
        pixel_masks, patch_labels = conversion
        numpy.savez_compressed(
            output_directory / f"{task_stem(label_task, task_index)}_labels.npz",
            pixel_masks=numpy.packbits(pixel_masks > 0, axis=-1),
            patch_labels=patch_labels,
            shape=numpy.array(pixel_masks.shape),
        )
        for class_name, class_mask in zip(SPOTTER_CLASSES, pixel_masks, strict=True):
            class_pixels[class_name] += int((class_mask > 0).sum())
        converted_count += 1
    return {"tasks": len(label_tasks), "converted": converted_count, "class_pixels": class_pixels}


def run_active_learning(settings: ReaderMarkSettings, task_count: int) -> dict:
    """Rank the unlabeled real pages and export the most uncertain ones.

    Usage:
        ``python -m readermarks active-learning``.

    Args:
        settings: Pipeline settings.
        task_count: Number of pages to export.

    Returns:
        ``{"export": path, "pages": ranking rows}``.
    """
    real_pages = load_real_pages(settings.path_settings.resolved(settings.path_settings.real_pages_directory))
    split_pages = load_or_create_split(
        real_pages,
        settings.path_settings.resolved(settings.path_settings.data_directory),
        settings.synthesis.random_seed,
    )
    candidate_pages = pages_of_split(real_pages, split_pages, "unlabeled")
    export_path = (
        settings.path_settings.resolved(settings.path_settings.data_directory)
        / ACTIVE_LEARNING_DIRECTORY
        / "label_studio_tasks.json"
    )
    page_cleaner = load_models(settings, compute_device()).page_cleaner
    selected_rows = export_uncertain_pages(
        page_cleaner, candidate_pages, task_count, export_path, picture_regions_of(settings)
    )
    return {"export": str(export_path), "pages": selected_rows}
