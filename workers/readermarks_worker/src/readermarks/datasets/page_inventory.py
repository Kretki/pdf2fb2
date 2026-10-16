"""Real page inventory: page-level labels and the locked train/val/test split.

The split is drawn once with a seeded shuffle and written to ``<data_directory>/splits.json``;
later runs reuse the file so the test pages never move (STAGE_CLEAN.md §12).
"""

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from readermarks.settings.model_settings import read_yaml_mapping

PAGE_LABELS_FILE = Path(__file__).resolve().parents[1] / "settings" / "real_page_labels.yaml"
PAGE_NAME_PATTERN = re.compile(r"^page_(\d{1,6})\.(png|jpg|jpeg|tif|tiff)$", re.IGNORECASE)
SPLITS_FILE_NAME = "splits.json"
SPLIT_NAMES = ("synthetic_bases", "real_validation", "real_test", "unlabeled")


@dataclass(frozen=True)
class RealPage:
    """One real page image with its page-level label.

    Attributes:
        page_number: 1-based page number parsed from ``page_NNNN.png``.
        image_path: Absolute image path.
        has_marks: True/False for labeled pages, None for unlabeled pages.
    """

    page_number: int
    image_path: Path
    has_marks: bool | None


def list_page_images(pages_directory: Path) -> dict[int, Path]:
    """Find ``page_NNNN.<ext>`` images in a directory.

    Usage:
        Call to enumerate the real pages; other files are ignored.

    Args:
        pages_directory: Directory with the page images.

    Returns:
        Mapping page number -> absolute image path, sorted by page number.
    """
    page_paths = {}
    for candidate_path in sorted(pages_directory.iterdir()):
        name_match = PAGE_NAME_PATTERN.match(candidate_path.name)
        if name_match and candidate_path.is_file():
            page_paths[int(name_match.group(1))] = candidate_path.resolve()
    return page_paths


def load_real_pages(pages_directory: Path) -> list[RealPage]:
    """Attach the page-level labels of ``real_page_labels.yaml`` to the page images.

    Args:
        pages_directory: Directory with the page images.

    Returns:
        All pages in page order; ``has_marks`` is None outside the labeled range.
    """
    label_mapping = read_yaml_mapping(PAGE_LABELS_FILE)
    marked_pages = set(label_mapping["marked_pages"])
    labeled_range = range(int(label_mapping["labeled_first"]), int(label_mapping["labeled_last"]) + 1)
    return [
        RealPage(page_number, image_path, (page_number in marked_pages) if page_number in labeled_range else None)
        for page_number, image_path in list_page_images(pages_directory).items()
    ]


def draw_split(real_pages: list[RealPage], random_seed: int) -> dict[str, list[int]]:
    """Assign labeled pages to splits with a seeded shuffle.

    Args:
        real_pages: Output of ``load_real_pages``.
        random_seed: Seed of the shuffle.

    Returns:
        Mapping split name -> sorted page numbers.

    Raises:
        ValueError: If there are fewer labeled pages than the configured split counts.
    """
    split_counts = read_yaml_mapping(PAGE_LABELS_FILE)["split_counts"]
    # B311/S311: reproducible split shuffle, not cryptography.
    random_source = random.Random(random_seed)  # nosec B311  # noqa: S311
    marked_numbers = [real_page.page_number for real_page in real_pages if real_page.has_marks is True]
    unmarked_numbers = [real_page.page_number for real_page in real_pages if real_page.has_marks is False]
    random_source.shuffle(marked_numbers)
    random_source.shuffle(unmarked_numbers)
    marked_cut = int(split_counts["marked_validation"])
    unmarked_cut = int(split_counts["unmarked_validation"])
    unmarked_end = unmarked_cut + int(split_counts["unmarked_test"])
    if len(marked_numbers) < marked_cut + int(split_counts["marked_test"]) or len(unmarked_numbers) <= unmarked_end:
        raise ValueError("Not enough labeled pages for the configured split counts.")
    split_pages = {
        "real_validation": marked_numbers[:marked_cut] + unmarked_numbers[:unmarked_cut],
        "real_test": marked_numbers[marked_cut:] + unmarked_numbers[unmarked_cut:unmarked_end],
        "synthetic_bases": unmarked_numbers[unmarked_end:],
        "unlabeled": [real_page.page_number for real_page in real_pages if real_page.has_marks is None],
    }
    return {split_name: sorted(page_numbers) for split_name, page_numbers in split_pages.items()}


def load_or_create_split(real_pages: list[RealPage], data_directory: Path, random_seed: int) -> dict[str, list[int]]:
    """Return the saved split, creating ``splits.json`` on the first call.

    Usage:
        Call from every command that needs the split; never edit the file by hand.

    Args:
        real_pages: Output of ``load_real_pages``.
        data_directory: Directory that holds ``splits.json``.
        random_seed: Seed used only when the file does not exist yet.

    Returns:
        Mapping split name -> sorted page numbers.
    """
    splits_file = data_directory / SPLITS_FILE_NAME
    if splits_file.is_file():
        return json.loads(splits_file.read_text(encoding="utf-8"))
    split_pages = draw_split(real_pages, random_seed)
    data_directory.mkdir(parents=True, exist_ok=True)
    splits_file.write_text(json.dumps(split_pages, indent=1), encoding="utf-8")
    return split_pages


def pages_of_split(real_pages: list[RealPage], split_pages: dict[str, list[int]], split_name: str) -> list[RealPage]:
    """Select the pages of one split.

    Args:
        real_pages: Output of ``load_real_pages``.
        split_pages: Output of ``load_or_create_split``.
        split_name: One of ``SPLIT_NAMES``.

    Returns:
        Pages of the split, in page order.

    Raises:
        KeyError: If the split name is unknown.
    """
    selected_numbers = set(split_pages[split_name])
    return [real_page for real_page in real_pages if real_page.page_number in selected_numbers]
