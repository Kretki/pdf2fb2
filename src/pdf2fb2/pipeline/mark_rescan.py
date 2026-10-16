"""Reader-mark rescan: clean marked pages, segment and OCR them again, merge with the first pass.

Flow (after the first layout + OCR pass):
    1. the page cleaner detects reader marks with picture segments blanked out;
    2. cleaned pages are copied to ``<output>/rescan/pages`` and segmented again;
    3. the new segments are OCR'd again (own cache, crops and job files under ``rescan/``);
    4. the rescan text is merged with the first-pass pictures; unmarked pages stay as they were.
"""

import shutil
from pathlib import Path

from pdf2fb2.pipeline.page_merging import merge_page_records
from pdf2fb2.pipeline.stage_contracts import PageCleaner, PageSegmenter, RescanProgress, SegmentRecognizer

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
    """Runs cleaning, re-segmentation and re-recognition of pages with reader marks.

    Usage:
        Built by the command line with the marks, layout and OCR stages; injected into the pipeline.
    """

    def __init__(self, page_cleaner: PageCleaner, page_segmenter: PageSegmenter,
                 segment_recognizer: SegmentRecognizer) -> None:
        """Store the stages.

        Args:
            page_cleaner: Reader-mark detection/removal stage.
            page_segmenter: Layout stage (also used for the first pass).
            segment_recognizer: OCR stage (also used for the first pass).
        """
        self._page_cleaner = page_cleaner
        self._page_segmenter = page_segmenter
        self._segment_recognizer = segment_recognizer

    def rescan_marked_pages(self, page_records: list[dict], work_directory: Path,
                            rescan_progress: RescanProgress) -> list[dict]:
        """Clean marked pages and replace their records by a rescan of the cleaned page.

        Args:
            page_records: OCR'd first-pass page records.
            work_directory: Conversion output directory.
            rescan_progress: Progress sinks of cleaning, re-segmentation and re-recognition.

        Returns:
            Final page records in page order.

        Raises:
            WorkerFailedError: If a model worker fails.
        """
        cleaned_pages = self._page_cleaner.clean_pages(page_records, work_directory,
                                                       rescan_progress.cleaning_progress)
        if not cleaned_pages:
            rescan_progress.segmentation_progress.advance_stage(1, 1, "no reader marks found")
            rescan_progress.recognition_progress.advance_stage(1, 1, "no pages to read again")
            return page_records
        rescan_directory = work_directory / RESCAN_DIRECTORY
        staged_images = stage_cleaned_pages(cleaned_pages, rescan_directory)
        rescanned_list = self._page_segmenter.segment_pages(staged_images, rescan_directory,
                                                            rescan_progress.segmentation_progress)
        rescanned_records = restore_page_numbers(rescanned_list, sorted(cleaned_pages))
        self._segment_recognizer.recognize_segments(list(rescanned_records.values()), rescan_directory,
                                                    rescan_progress.recognition_progress)
        return merge_page_records(page_records, rescanned_records)
