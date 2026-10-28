"""Conversion results: segmentation drawings and the end-of-run summary."""

from dataclasses import dataclass
from pathlib import Path

from pdf2fb2.parallel_work.parallel_mapping import map_in_threads
from pdf2fb2.progress.progress_contracts import StageProgress
from pdf2fb2.rendering.segment_drawing import draw_segment_overlay

VISUALIZATION_DIRECTORY = "vis"


@dataclass(frozen=True)
class ConversionOutcome:
    """What a finished conversion produced.

    Attributes:
        fb2_path: Written FB2 file.
        book_statistics: Element counts of the FB2.
        detected_problems: Problems found by the segmentation and FB2 checks.
        drawings_directory: Segmentation drawings directory, or None when disabled.
    """

    fb2_path: Path
    book_statistics: dict
    detected_problems: list[str]
    drawings_directory: Path | None


def draw_overlays(page_records: list[dict], output_directory: Path, stage_progress: StageProgress) -> Path:
    """Save segmentation drawings for every page.

    Args:
        page_records: Segmented pages.
        output_directory: Conversion output directory.
        stage_progress: Receives ``drawing segments of page k/n`` updates.

    Returns:
        Directory containing the drawings.
    """
    drawings_directory = output_directory / VISUALIZATION_DIRECTORY
    drawings_directory.mkdir(exist_ok=True)
    map_in_threads(lambda page_record: draw_segment_overlay(
        Path(page_record["image"]), page_record["segments"],
        drawings_directory / f"page_{page_record['page']:04d}_segments.png"), page_records,
        lambda drawn_count, page_count: stage_progress.advance_stage(
            drawn_count, page_count, f"drawing segments of page {drawn_count}/{page_count}"))
    return drawings_directory


def report_outcome(conversion_outcome: ConversionOutcome) -> int:
    """Print the conversion summary.

    Args:
        conversion_outcome: Result of the conversion.

    Returns:
        Exit code: 0 when all checks passed, 1 otherwise.
    """
    print(f"FB2: {conversion_outcome.fb2_path}  {conversion_outcome.book_statistics}")
    if conversion_outcome.drawings_directory is not None:
        print(f"Segment drawings: {conversion_outcome.drawings_directory}")
    if not conversion_outcome.detected_problems:
        print("All checks passed.")
        return 0
    print("\nCHECKS FAILED:")
    for detected_problem in conversion_outcome.detected_problems:
        print("  -", detected_problem)
    return 1
