"""High-level evaluation actions used by the CLI: threshold tuning and the §9 report."""

import os
from pathlib import Path

from readermarks.datasets.layout_regions import picture_regions_of
from readermarks.datasets.page_inventory import load_or_create_split, load_real_pages, pages_of_split
from readermarks.evaluation.page_evaluation import evaluate_real_pages
from readermarks.evaluation.patch_evaluation import evaluate_patches, free_cuda_cache
from readermarks.evaluation.pixel_evaluation import evaluate_synthetic_pixels
from readermarks.evaluation.report_writer import write_report
from readermarks.evaluation.threshold_tuning import tune_thresholds
from readermarks.inference.model_loading import load_models
from readermarks.networks.checkpoint_store import save_thresholds
from readermarks.settings.model_settings import ReaderMarkSettings
from readermarks.synthesis.synthesis_service import synthetic_split_directory
from readermarks.training.training_service import compute_device, feature_directory

REAL_SPLITS = {"real_val": "real_validation", "real_test": "real_test"}
REPORT_DIRECTORY = "reports"
TEXTAR_VARIABLE = "READERMARKS_TEXTAR_DIR"
GROUND_TRUTH_VARIABLE = "READERMARKS_GT_TEXT_DIR"
SYNTHETIC_PIXEL_SAMPLES = 200


def labeled_pages(settings: ReaderMarkSettings, split_name: str) -> list:
    """Real pages of a split.

    Args:
        settings: Pipeline settings.
        split_name: Key of ``REAL_SPLITS`` values (``real_validation`` / ``real_test``).

    Returns:
        Real pages.
    """
    real_pages = load_real_pages(settings.path_settings.resolved(settings.path_settings.real_pages_directory))
    split_pages = load_or_create_split(
        real_pages,
        settings.path_settings.resolved(settings.path_settings.data_directory),
        settings.synthesis.random_seed,
    )
    return pages_of_split(real_pages, split_pages, split_name)


def run_tuning(settings: ReaderMarkSettings) -> dict:
    """Tune thresholds on ``real_validation`` and save them to ``thresholds.json``.

    Usage:
        ``python -m readermarks tune``.

    Args:
        settings: Pipeline settings.

    Returns:
        Tuning summary.
    """
    loaded_models = load_models(settings, compute_device())
    tuning_result = tune_thresholds(
        loaded_models.page_cleaner,
        loaded_models.spotter_runner,
        loaded_models.segmenter_runner,
        labeled_pages(settings, "real_validation"),
        picture_regions_of(settings),
    )
    save_thresholds(
        settings.path_settings.resolved(settings.path_settings.checkpoint_directory), tuning_result["thresholds"]
    )
    return tuning_result


def downstream_notes() -> list[str]:
    """Notes about the downstream hooks (TexTAR, OCR CER) that need external resources.

    Returns:
        Human-readable notes.
    """
    note_lines = []
    if not os.environ.get(TEXTAR_VARIABLE):
        note_lines.append(f"WARNING: TexTAR false-underline metric skipped ({TEXTAR_VARIABLE} not set).")
    if not os.environ.get(GROUND_TRUTH_VARIABLE):
        note_lines.append(f"WARNING: OCR CER metric skipped ({GROUND_TRUTH_VARIABLE} not set, no ground-truth text).")
    note_lines.append(
        "Real pages have page-level labels only: patch/pixel metrics are measured on the synthetic test "
        "split; the real pixel-masked set is not available."
    )
    return note_lines


def run_evaluation(settings: ReaderMarkSettings, split_key: str) -> tuple[Path, Path, dict]:
    """Produce the §9 report for ``real_val`` or ``real_test``.

    Usage:
        ``python -m readermarks eval --split real_val``.

    Args:
        settings: Pipeline settings.
        split_key: ``real_val`` or ``real_test``.

    Returns:
        ``(json_path, markdown_path, report)``.

    Raises:
        KeyError: If the split key is unknown.
    """
    loaded_models = load_models(settings, compute_device())
    cleaning_settings = loaded_models.page_cleaner.cleaning_settings
    report_document = {"split": split_key, "thresholds": cleaning_settings.model_dump(), "notes": downstream_notes()}
    report_document["synthetic_patches"] = evaluate_patches(
        loaded_models.spotter_runner, feature_directory(settings, "test"), cleaning_settings.patch_threshold
    )
    report_document["synthetic_pixels"] = evaluate_synthetic_pixels(
        loaded_models.page_cleaner, synthetic_split_directory(settings, "test"), SYNTHETIC_PIXEL_SAMPLES
    )
    free_cuda_cache()
    report_document["real_pages"] = evaluate_real_pages(
        loaded_models.page_cleaner, labeled_pages(settings, REAL_SPLITS[split_key]), picture_regions_of(settings)
    )
    report_directory = settings.path_settings.resolved(settings.path_settings.data_directory) / REPORT_DIRECTORY
    json_path, markdown_path = write_report(report_document, report_directory)
    return json_path, markdown_path, report_document
