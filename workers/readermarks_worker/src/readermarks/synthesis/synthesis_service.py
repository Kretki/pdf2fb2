"""High-level synthetic-data actions used by the CLI: generate all splits, render previews."""

from pathlib import Path

from readermarks.datasets.page_inventory import load_or_create_split, load_real_pages, pages_of_split
from readermarks.settings.model_settings import ReaderMarkSettings
from readermarks.synthesis.dataset_writer import SplitRequest, default_worker_count, read_manifest, write_split
from readermarks.synthesis.preview_sheet import write_preview_sheets

SYNTHETIC_DIRECTORY = "synthetic"
PREVIEW_DIRECTORY = "previews"


def synthetic_split_directory(settings: ReaderMarkSettings, split_name: str) -> Path:
    """Directory of a generated synthetic split.

    Args:
        settings: Pipeline settings.
        split_name: ``train``, ``validation`` or ``test``.

    Returns:
        ``<data_directory>/synthetic/<split_name>``.
    """
    return settings.path_settings.resolved(settings.path_settings.data_directory) / SYNTHETIC_DIRECTORY / split_name


def base_pages_by_split(settings: ReaderMarkSettings) -> dict[str, tuple[Path, ...]]:
    """Clean base pages of each synthetic split.

    The training split uses the ``synthetic_bases`` pages; validation and test use the unmarked
    pages of ``real_validation`` so that synthetic evaluation never sees a training base page.

    Args:
        settings: Pipeline settings.

    Returns:
        Mapping split name -> base page paths.
    """
    real_pages = load_real_pages(settings.path_settings.resolved(settings.path_settings.real_pages_directory))
    split_pages = load_or_create_split(
        real_pages,
        settings.path_settings.resolved(settings.path_settings.data_directory),
        settings.synthesis.random_seed,
    )
    training_bases = tuple(
        real_page.image_path for real_page in pages_of_split(real_pages, split_pages, "synthetic_bases")
    )
    holdout_bases = tuple(
        real_page.image_path
        for real_page in pages_of_split(real_pages, split_pages, "real_validation")
        if real_page.has_marks is False
    )
    return {"train": training_bases, "validation": holdout_bases, "test": holdout_bases}


def generate_synthetic_dataset(settings: ReaderMarkSettings, split_names: tuple[str, ...]) -> dict[str, Path]:
    """Generate the requested synthetic splits.

    Usage:
        ``python -m readermarks synth``.

    Args:
        settings: Pipeline settings.
        split_names: Subset of ``("train", "validation", "test")``.

    Returns:
        Mapping split name -> manifest path.
    """
    sample_counts = {
        "train": settings.synthesis.train_count,
        "validation": settings.synthesis.validation_count,
        "test": settings.synthesis.test_count,
    }
    split_bases = base_pages_by_split(settings)
    manifest_paths = {}
    for split_name in split_names:
        split_request = SplitRequest(
            split_name,
            split_bases[split_name],
            sample_counts[split_name],
            synthetic_split_directory(settings, split_name),
        )
        manifest_paths[split_name] = write_split(split_request, settings.synthesis, default_worker_count())
    return manifest_paths


def render_previews(settings: ReaderMarkSettings, split_name: str) -> list[Path]:
    """Render preview sheets of the first ``preview_count`` samples of a split.

    Usage:
        ``python -m readermarks preview``.

    Args:
        settings: Pipeline settings.
        split_name: Split to preview.

    Returns:
        Paths of the preview sheets.
    """
    split_directory = synthetic_split_directory(settings, split_name)
    sample_names = [manifest_record["sample"] for manifest_record in read_manifest(split_directory)][
        : settings.synthesis.preview_count
    ]
    preview_directory = (
        settings.path_settings.resolved(settings.path_settings.data_directory) / PREVIEW_DIRECTORY / split_name
    )
    return write_preview_sheets(split_directory, sample_names, preview_directory)
