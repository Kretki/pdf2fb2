"""Write the per-page outputs: clean image, mark masks, ``marks.json`` and the debug overlay.

Unmarked pages are copied byte for byte (``clean<original suffix>``), so they are never altered.
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy

from readermarks.datasets.image_files import write_mask_image, write_rgb_image
from readermarks.inference.debug_overlay import build_debug_overlay
from readermarks.inference.page_cleaner import PageVerdict
from readermarks.postprocess.mark_records import build_mark_records
from readermarks.settings.class_catalog import MARK_CLASSES

MARKS_FILE = "marks.json"
MARK_MASK_FILE = "marks_mask.png"
CLASS_MASK_DIRECTORY = "masks"
DEBUG_FILE = "debug.png"


@dataclass(frozen=True)
class WrittenPage:
    """Where the outputs of one page went.

    Attributes:
        clean_path: Clean page image (byte copy of the input for unmarked pages).
        marks_path: ``marks.json``.
        has_marks: Page-level flag.
        mark_count: Number of kept marks.
    """

    clean_path: Path
    marks_path: Path
    has_marks: bool
    mark_count: int


def marks_document(input_path: Path, page_verdict: PageVerdict) -> dict:
    """JSON document of ``marks.json``.

    Args:
        input_path: Source page image.
        page_verdict: Cleaner verdict.

    Returns:
        ``{"image", "has_marks", "page_probability", "regions", "marks", "timings"}``.
    """
    mark_records = build_mark_records(page_verdict.fused_marks) if page_verdict.has_marks else []
    return {
        "image": input_path.name,
        "has_marks": page_verdict.has_marks,
        "page_probability": round(page_verdict.spotter_output.page_probability, 4),
        "regions": [list(region_box) for region_box in page_verdict.region_boxes],
        "marks": mark_records,
        "timings": {stage_name: round(stage_time, 4) for stage_name, stage_time in page_verdict.stage_seconds.items()},
    }


def write_masks(output_directory: Path, page_verdict: PageVerdict) -> None:
    """Write ``marks_mask.png`` (any mark) and one mask per class under ``masks/``.

    Args:
        output_directory: Page output directory.
        page_verdict: Verdict of a marked page.

    Returns:
        None.
    """
    fused_marks = page_verdict.fused_marks
    write_mask_image(output_directory / MARK_MASK_FILE, fused_marks.mark_mask.astype(numpy.uint8) * 255)
    for class_name, class_mask in zip(MARK_CLASSES, fused_marks.class_masks, strict=True):
        write_mask_image(
            output_directory / CLASS_MASK_DIRECTORY / f"{class_name}.png", class_mask.astype(numpy.uint8) * 255
        )
    write_mask_image(
        output_directory / CLASS_MASK_DIRECTORY / "printed_ink.png", fused_marks.ink_mask.astype(numpy.uint8) * 255
    )
    write_mask_image(
        output_directory / CLASS_MASK_DIRECTORY / "protected_rules.png",
        fused_marks.protected_mask.astype(numpy.uint8) * 255,
    )


def write_page_outputs(
    input_path: Path,
    rgb_image: numpy.ndarray,
    page_verdict: PageVerdict,
    output_directory: Path,
    write_debug: bool,
    patch_threshold: float,
) -> WrittenPage:
    """Write all outputs of one page.

    Usage:
        Called by the ``clean`` command and by the pdf2fb2 worker.

    Args:
        input_path: Source page image.
        rgb_image: Decoded page (for the debug overlay).
        page_verdict: Cleaner verdict.
        output_directory: Page output directory (created).
        write_debug: Also write ``debug.png``.
        patch_threshold: Patch threshold drawn in the overlay grid.

    Returns:
        Paths and flags of the written page.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    if page_verdict.has_marks:
        clean_path = output_directory / "clean.png"
        write_rgb_image(clean_path, page_verdict.cleaned_image)
        write_masks(output_directory, page_verdict)
    else:
        clean_path = output_directory / f"clean{input_path.suffix.lower()}"
        shutil.copyfile(input_path, clean_path)
    if write_debug:
        write_rgb_image(
            output_directory / DEBUG_FILE,
            build_debug_overlay(rgb_image, page_verdict.spotter_output, page_verdict.fused_marks, patch_threshold),
        )
    document = marks_document(input_path, page_verdict)
    marks_path = output_directory / MARKS_FILE
    marks_path.write_text(json.dumps(document, indent=1), encoding="utf-8")
    return WrittenPage(clean_path, marks_path, page_verdict.has_marks, len(document["marks"]))
