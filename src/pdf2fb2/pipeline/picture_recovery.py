"""Picture recovery: lettering that does not fit the book's font size is a picture the layout model took for text.

Flow (after the reader-mark rescan, before OCR):
    1. glyph heights of every segment are measured with OpenCV and stored as ``glyph_height``; their median
       over the plain paragraphs of the book is the usual font size (see ``segments.font_statistics``);
    2. pages with off-size text or with large lettering outside every segment are segmented again with a
       low layout threshold (job files under ``<output>/picture_rescan``);
    3. the most probable picture proposal of each region replaces the text segments it covers and the
       off-size text next to it (see ``segments.picture_selection``);
    4. the font statistics and every region with its outcome are written to ``picture_rescan/font_statistics.json``.
"""

from dataclasses import asdict
from pathlib import Path

from pdf2fb2.parallel_work.parallel_mapping import map_in_threads
from pdf2fb2.pipeline.mark_rescan import stage_cleaned_pages
from pdf2fb2.pipeline.stage_contracts import BoxProposer, GlyphMeter
from pdf2fb2.progress.progress_contracts import StageProgress
from pdf2fb2.segments.font_statistics import (
    FontStatistics,
    OffSizeRegion,
    PageGlyphs,
    collect_font_statistics,
    find_off_size_regions,
    store_glyph_heights,
)
from pdf2fb2.segments.picture_selection import PictureChoice, apply_picture, choose_picture, is_covered
from worker_protocol.json_files import write_json_document

PICTURE_RESCAN_DIRECTORY = "picture_rescan"
FONT_REPORT_FILE = "font_statistics.json"


def region_report(off_size_region: OffSizeRegion, picture_choice: PictureChoice | None) -> dict:
    """JSON description of one off-size region and its outcome.

    Args:
        off_size_region: The region.
        picture_choice: Picture that replaced it, or None when it was kept.

    Returns:
        Dict with the page, reason, region box and the picture box and score (None when kept).
    """
    return {"page": off_size_region.page_record["page"], "reason": off_size_region.region_reason,
            "region_box": [round(coordinate, 1) for coordinate in off_size_region.region_box],
            "picture_box": None if picture_choice is None else picture_choice.picture_box,
            "picture_score": None if picture_choice is None else round(picture_choice.picture_score, 4)}


def write_font_report(rescan_directory: Path, font_statistics: FontStatistics,
                      region_outcomes: list[tuple[OffSizeRegion, PictureChoice | None]]) -> None:
    """Write the font statistics and the off-size regions with their outcome.

    Args:
        rescan_directory: ``<output>/picture_rescan`` (created when missing).
        font_statistics: Usual font size of the book.
        region_outcomes: Every region with the picture that replaced it, or None.

    Returns:
        None. Creates or replaces ``font_statistics.json``.

    Raises:
        OSError: If the file cannot be written.
    """
    rescan_directory.mkdir(parents=True, exist_ok=True)
    write_json_document(rescan_directory / FONT_REPORT_FILE, {
        "font_statistics": asdict(font_statistics),
        "regions": [region_report(off_size_region, picture_choice)
                    for off_size_region, picture_choice in region_outcomes]})


def unique_region_pages(off_size_regions: list[OffSizeRegion]) -> list[dict]:
    """Page records of the regions, each once, in page order.

    Args:
        off_size_regions: Regions in page order.

    Returns:
        Distinct page records.
    """
    region_pages: dict[int, dict] = {}
    for off_size_region in off_size_regions:
        region_pages.setdefault(off_size_region.page_record["page"], off_size_region.page_record)
    return [region_pages[page_number] for page_number in sorted(region_pages)]


class PictureRecovery:
    """Replaces off-size lettering by the picture block the layout model finds most probable for it.

    Usage:
        Built by the command line with the OpenCV glyph meter and a low-threshold layout stage; injected
        into the pipeline and run on the final segmented pages before OCR.
    """

    def __init__(self, glyph_meter: GlyphMeter, box_proposer: BoxProposer) -> None:
        """Store the measurement and the layout proposals.

        Args:
            glyph_meter: Measures the lettering of a page.
            box_proposer: Layout model run with a low score threshold.
        """
        self._glyph_meter = glyph_meter
        self._box_proposer = box_proposer

    def recover_pictures(self, page_records: list[dict], work_directory: Path,
                         stage_progress: StageProgress) -> int:
        """Measure the lettering, find off-size regions and replace them by pictures.

        Args:
            page_records: Final segmented page records, not yet OCR'd (mutated: ``glyph_height`` stored,
                recovered pictures replace text segments).
            work_directory: Conversion output directory.
            stage_progress: Receives the step's progress.

        Returns:
            Number of recovered pictures.

        Raises:
            WorkerFailedError: If the layout worker fails.
            ValueError: If a page image cannot be read.
        """
        page_measurements = self._measure_pages(page_records, stage_progress)
        font_statistics = collect_font_statistics(page_records)
        if font_statistics is None:
            stage_progress.advance_stage(1, 1, "too few paragraphs for font statistics")
            return 0
        off_size_regions = find_off_size_regions(page_records, page_measurements, font_statistics)
        picture_choices = self._choose_pictures(off_size_regions, font_statistics, work_directory, stage_progress)
        write_font_report(work_directory / PICTURE_RESCAN_DIRECTORY, font_statistics,
                          list(zip(off_size_regions, picture_choices, strict=True)))
        recovered_count = sum(picture_choice is not None for picture_choice in picture_choices)
        stage_progress.advance_stage(1, 1, f"usual glyph height {font_statistics.typical_height:g} px, "
                                           f"{recovered_count}/{len(off_size_regions)} off-size regions are pictures")
        return recovered_count

    def _measure_pages(self, page_records: list[dict], stage_progress: StageProgress) -> list[PageGlyphs]:
        """Measure every page and store the segment glyph heights.

        Args:
            page_records: Page records (mutated: ``glyph_height`` stored).
            stage_progress: Receives ``measuring fonts of page k/n`` updates.

        Returns:
            Measurements aligned with ``page_records``.
        """
        page_measurements = map_in_threads(
            self._glyph_meter.measure_page, page_records,
            lambda measured_count, page_count: stage_progress.advance_stage(
                measured_count, 2 * page_count, f"measuring fonts of page {measured_count}/{page_count}"))
        for page_record, page_glyphs in zip(page_records, page_measurements, strict=True):
            store_glyph_heights(page_record, page_glyphs)
        return page_measurements

    def _choose_pictures(self, off_size_regions: list[OffSizeRegion], font_statistics: FontStatistics,
                         work_directory: Path, stage_progress: StageProgress) -> list[PictureChoice | None]:
        """Segment the region pages again and replace each region by its most probable picture.

        Args:
            off_size_regions: Regions in page order (their page records are mutated).
            font_statistics: Usual font size of the book.
            work_directory: Conversion output directory.
            stage_progress: Receives the layout progress.

        Returns:
            Per region, the applied picture or None when it was kept as it is.
        """
        if not off_size_regions:
            return []
        proposals_by_page = self._propose_boxes(unique_region_pages(off_size_regions), work_directory,
                                                stage_progress)
        recovered_boxes: dict[int, list[list[float]]] = {}
        picture_choices = []
        for off_size_region in off_size_regions:
            page_number = off_size_region.page_record["page"]
            page_pictures = recovered_boxes.setdefault(page_number, [])
            picture_choice = None
            if not any(is_covered(off_size_region.region_box, picture_box) for picture_box in page_pictures):
                picture_choice = choose_picture(off_size_region, proposals_by_page[page_number], font_statistics)
            if picture_choice is not None:
                apply_picture(off_size_region.page_record, picture_choice)
                page_pictures.append(picture_choice.picture_box)
            picture_choices.append(picture_choice)
        return picture_choices

    def _propose_boxes(self, region_pages: list[dict], work_directory: Path,
                       stage_progress: StageProgress) -> dict[int, list[dict]]:
        """Run the low-threshold layout on copies of the region pages.

        Args:
            region_pages: Page records to segment again, in page order.
            work_directory: Conversion output directory.
            stage_progress: Receives the layout progress.

        Returns:
            Mapping page number -> proposed boxes.
        """
        rescan_directory = work_directory / PICTURE_RESCAN_DIRECTORY
        staged_images = stage_cleaned_pages({page_record["page"]: Path(page_record["image"])
                                             for page_record in region_pages}, rescan_directory)
        page_proposals = self._box_proposer.propose_boxes(staged_images, rescan_directory, stage_progress)
        return dict(zip([page_record["page"] for page_record in region_pages], page_proposals, strict=True))
