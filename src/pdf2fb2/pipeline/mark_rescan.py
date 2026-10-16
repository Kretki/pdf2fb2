"""Reader-mark rescan: clean marked pages, segment them again, merge with the first layout pass.

Flow (after the first layout pass, before any OCR):
    1. the page cleaner detects reader marks with picture segments blanked out;
    2. cleaned pages are copied to ``<output>/rescan/pages`` and segmented again (job files under ``rescan/``);
    3. the rescan text segments are merged with the first-pass pictures; unmarked pages stay as they were.
The merged records are read by OCR once, afterwards, by the conversion pipeline.
"""

import shutil
from pathlib import Path

from pdf2fb2.pipeline.page_merging import merge_page_records
from pdf2fb2.pipeline.stage_contracts import PageCleaner, PageSegmenter, RescanProgress

RESCAN_DIRECTORY = "rescan"
RESCAN_PAGES_DIRECTORY = "pages"


def stage_cleaned_pages(cleaned_pages: dict[int, Path], rescan_directory: Path) -> list[Path]:
    """Copy cleaned pages into the rescan working directory as ``page_NNNN.png``.

    Args:
        cleaned_pages: Mapping page number -> cleaned image.
        rescan_directory: ``<output>/rescan``.

    Returns:
        Copied images in ascending page order.
    """
    pages_directory = rescan_directory / RESCAN_PAGES_DIRECTORY
    pages_directory.mkdir(parents=True, exist_ok=True)
    staged_images = []
    for page_number in sorted(cleaned_pages):
        staged_image = pages_directory / f"page_{page_number:04d}{cleaned_pages[page_number].suffix}"
        shutil.copyfile(cleaned_pages[page_number], staged_image)
        staged_images.append(staged_image)
    return staged_images


def restore_page_numbers(rescanned_records: list[dict], page_numbers: list[int]) -> dict[int, dict]:
    """Give the rescanned records (numbered 1..n by the segmenter) their book page numbers.

    Args:
        rescanned_records: Records from the segmenter, in staged order (mutated: ``page`` rewritten).
        page_numbers: Original page numbers in the same order.

    Returns:
        Mapping page number -> rescanned record.
    """
    for rescanned_record, page_number in zip(rescanned_records, page_numbers, strict=True):
        rescanned_record["page"] = page_number
    return {rescanned_record["page"]: rescanned_record for rescanned_record in rescanned_records}


class MarkRescanner:
    """Runs cleaning and re-segmentation of pages with reader marks (no OCR).

    Usage:
        Built by the command line with the marks and layout stages; injected into the pipeline.
    """

    def __init__(self, page_cleaner: PageCleaner, page_segmenter: PageSegmenter) -> None:
        """Store the stages.

        Args:
            page_cleaner: Reader-mark detection/removal stage.
            page_segmenter: Layout stage (also used for the first pass).
        """
        self._page_cleaner = page_cleaner
        self._page_segmenter = page_segmenter

    def rescan_marked_pages(self, page_records: list[dict], work_directory: Path,
                            rescan_progress: RescanProgress) -> list[dict]:
        """Clean marked pages and replace their records by a re-segmentation of the cleaned page.

        Args:
            page_records: Segmented (not yet OCR'd) first-pass page records.
            work_directory: Conversion output directory.
            rescan_progress: Progress sinks of cleaning and re-segmentation.

        Returns:
            Final segmented page records in page order, ready for OCR.

        Raises:
            WorkerFailedError: If a model worker fails.
        """
        cleaned_pages = self._page_cleaner.clean_pages(page_records, work_directory,
                                                       rescan_progress.cleaning_progress)
        if not cleaned_pages:
            rescan_progress.segmentation_progress.advance_stage(1, 1, "no reader marks found")
            return page_records
        rescan_directory = work_directory / RESCAN_DIRECTORY
        staged_images = stage_cleaned_pages(cleaned_pages, rescan_directory)
        rescanned_list = self._page_segmenter.segment_pages(staged_images, rescan_directory,
                                                            rescan_progress.segmentation_progress)
        rescanned_records = restore_page_numbers(rescanned_list, sorted(cleaned_pages))
        return merge_page_records(page_records, rescanned_records)
