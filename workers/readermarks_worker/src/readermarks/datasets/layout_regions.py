"""Picture regions of real pages from a pdf2fb2 ``segments.json`` (PP-DocLayoutV3 output).

Evaluation and tuning blank these regions exactly as the pdf2fb2 pipeline does before the
reader-mark stage runs, so measured accuracy matches production behaviour.
"""

import json
from pathlib import Path

import numpy

from readermarks.datasets.image_files import read_rgb_image
from readermarks.datasets.page_inventory import RealPage
from readermarks.inference.region_masking import blank_regions
from readermarks.settings.model_settings import ReaderMarkSettings

# Must match PICTURE_LABELS in pdf2fb2.segments.label_roles (labels embedded as pictures).
PICTURE_LABELS = frozenset({"image", "chart", "seal", "display_formula"})
MAX_SEGMENTS_BYTES = 256 * 1024 * 1024


def load_picture_regions(segments_file: Path | None) -> dict[int, list[list[float]]]:
    """Picture boxes per page number.

    Args:
        segments_file: pdf2fb2 ``segments.json`` (list of page records), or None.

    Returns:
        Mapping page number -> list of ``[x_start, y_start, x_end, y_end]``; empty without a file.

    Raises:
        ValueError: If the file is too large or not a list of page records.
    """
    if segments_file is None or not segments_file.is_file():
        return {}
    if segments_file.stat().st_size > MAX_SEGMENTS_BYTES:
        raise ValueError(f"{segments_file} is larger than {MAX_SEGMENTS_BYTES} bytes.")
    page_records = json.loads(segments_file.read_text(encoding="utf-8"))
    if not isinstance(page_records, list):
        raise ValueError(f"{segments_file} must contain a list of page records.")
    return {
        int(page_record["page"]): [
            [float(coordinate) for coordinate in page_segment["bbox"]]
            for page_segment in page_record.get("segments", [])
            if page_segment.get("label") in PICTURE_LABELS
        ]
        for page_record in page_records
        if isinstance(page_record, dict) and "page" in page_record
    }


def read_masked_page(real_page: RealPage, picture_regions: dict[int, list[list[float]]]) -> numpy.ndarray:
    """Read a real page with its picture regions blanked.

    Usage:
        Used by evaluation, threshold tuning and active learning.

    Args:
        real_page: Page to read.
        picture_regions: Output of ``load_picture_regions``.

    Returns:
        (H, W, 3) uint8 page.
    """
    return blank_regions(read_rgb_image(real_page.image_path), picture_regions.get(real_page.page_number, []))


def picture_regions_of(settings: ReaderMarkSettings) -> dict[int, list[list[float]]]:
    """Picture boxes of the real pages from the configured layout segments file.

    Args:
        settings: Pipeline settings.

    Returns:
        Mapping page number -> boxes (empty when no file is configured).
    """
    segments_file = settings.path_settings.layout_segments_file
    return load_picture_regions(None if segments_file is None else settings.path_settings.resolved(segments_file))
