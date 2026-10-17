"""Residual-text rescan: after cleaning and picture recovery, text left outside every segment is segmented again.

Flow (last step before OCR):
    1. every page is measured with all its segments blanked; letters of the usual font size that remain are
       grouped into lines and blocks (see ``segments.residual_text``);
    2. missed running heads and page numbers (in the bands where the book prints them) are added as found;
    3. pages with missed paragraphs inside the printed frame are segmented again (job files under
       ``<output>/residual_rescan``) and each paragraph gets the layout model's proposal for it, or its own box
       when the model has none (see ``segments.residual_segments``);
    4. the blocks and their segments are written to ``residual_rescan/residual_text.json``.
"""

from dataclasses import asdict, replace
from pathlib import Path

from pdf2fb2.pipeline.mark_rescan import stage_cleaned_pages
from pdf2fb2.pipeline.page_merging import insert_by_position
from pdf2fb2.pipeline.stage_contracts import BoxProposer, GlyphMeter
from pdf2fb2.progress.progress_contracts import StageProgress
from pdf2fb2.segments.font_statistics import (
    FontStatistics,
    PageGlyphs,
    collect_font_statistics,
    is_inside_frame,
    store_glyph_heights,
)
from pdf2fb2.segments.frame_statistics import PageFrame, group_frames
from pdf2fb2.segments.label_roles import PICTURE_LABELS
from pdf2fb2.segments.residual_segments import residual_segment
from pdf2fb2.segments.residual_text import (
    TEXT_LABEL,
    ResidualBlock,
    collect_margin_bands,
    find_residual_blocks,
    overlaps_horizontally,
    usual_glyphs,
)
from worker_protocol.json_files import write_json_document

RESIDUAL_DIRECTORY = "residual_rescan"
RESIDUAL_REPORT_FILE = "residual_text.json"
BlockSegments = list[tuple[ResidualBlock, dict]]  # residual blocks with the segments made from them
CAPTION_LABEL = "figure_title"
CAPTION_GAP_FACTOR = 3.0  # largest gap between a picture and its caption, in usual glyph heights


def is_caption_block(residual_block: ResidualBlock, page_segments: list[dict], typical_height: float) -> bool:
    """Tell whether a missed paragraph sits right below a picture (its caption).

    Args:
        residual_block: Missed paragraph.
        page_segments: Segments of the page.
        typical_height: Usual glyph height in pixels.

    Returns:
        True if a picture segment ends at most ``CAPTION_GAP_FACTOR`` glyph heights above the block and overlaps
        it horizontally.
    """
    block_box = residual_block.block_box
    return any(page_segment["label"] in PICTURE_LABELS and overlaps_horizontally(page_segment["bbox"], block_box)
               and 0 <= block_box[1] - page_segment["bbox"][3] <= CAPTION_GAP_FACTOR * typical_height
               for page_segment in page_segments)


def classify_block(residual_block: ResidualBlock, page_record: dict, page_frame: PageFrame | None,
                   font_statistics: FontStatistics) -> ResidualBlock | None:
    """Keep, relabel or drop one residual block.

    Args:
        residual_block: Block found on the page.
        page_record: Its page record.
        page_frame: Printed frame of the page group, or None.
        font_statistics: Usual font size of the book.

    Returns:
        Band elements unchanged; paragraphs right below a picture as captions; other paragraphs only when inside
        the printed frame (outside it they are reader notes): None for dropped blocks.
    """
    if residual_block.block_label != TEXT_LABEL:
        return residual_block
    if is_caption_block(residual_block, page_record["segments"], font_statistics.typical_height):
        return replace(residual_block, block_label=CAPTION_LABEL)
    if is_inside_frame(residual_block.block_box, float(page_record["size"][1]), page_frame):
        return residual_block
    return None


def page_residual_blocks(page_records: list[dict], page_measurements: list[PageGlyphs],
                         font_statistics: FontStatistics) -> dict[int, list[ResidualBlock]]:
    """Residual blocks of every page, classified by ``classify_block``.

    Args:
        page_records: Final segmented page records.
        page_measurements: Measurements aligned with ``page_records``.
        font_statistics: Usual font size of the book.

    Returns:
        Mapping page number -> blocks, only for pages that have some.
    """
    frame_by_parity = group_frames(page_records)
    bands_by_parity = collect_margin_bands(page_records)
    blocks_by_page = {}
    for page_record, page_glyphs in zip(page_records, page_measurements, strict=True):
        page_parity = page_record["page"] % 2
        found_blocks = find_residual_blocks(page_record, usual_glyphs(list(page_glyphs.uncovered_glyphs),
                                                                      font_statistics),
                                            font_statistics, bands_by_parity.get(page_parity))
        classified_blocks = [classify_block(residual_block, page_record, frame_by_parity.get(page_parity),
                                            font_statistics) for residual_block in found_blocks]
        residual_blocks = [residual_block for residual_block in classified_blocks if residual_block is not None]
        if residual_blocks:
            blocks_by_page[page_record["page"]] = residual_blocks
    return blocks_by_page


def write_residual_report(residual_directory: Path, added_segments: dict[int, BlockSegments]) -> None:
    """Write the residual blocks and the segments made from them.

    Args:
        residual_directory: ``<output>/residual_rescan`` (created when missing).
        added_segments: Mapping page number -> (block, segment) pairs.

    Returns:
        None. Creates or replaces ``residual_text.json``.

    Raises:
        OSError: If the file cannot be written.
    """
    residual_directory.mkdir(parents=True, exist_ok=True)
    write_json_document(residual_directory / RESIDUAL_REPORT_FILE, {"pages": [
        {"page": page_number, "blocks": [{**asdict(residual_block), "segment": dict(new_segment)}
                                          for residual_block, new_segment in block_segments]}
        for page_number, block_segments in sorted(added_segments.items())]})


class ResidualTextRescan:
    """Adds the text the layout passes missed, found as letters of the usual size outside every segment.

    Usage:
        Built by the command line with the OpenCV glyph meter and a layout stage; injected into the pipeline and
        run on the final segmented pages, after the picture recovery and before OCR.
    """

    def __init__(self, glyph_meter: GlyphMeter, box_proposer: BoxProposer) -> None:
        """Store the measurement and the layout proposals.

        Args:
            glyph_meter: Measures the lettering of a page.
            box_proposer: Layout model used to segment the pages with missed paragraphs again.
        """
        self._glyph_meter = glyph_meter
        self._box_proposer = box_proposer

    def add_residual_text(self, page_records: list[dict], work_directory: Path, stage_progress: StageProgress) -> int:
        """Find residual text on every page and add segments for it.

        Args:
            page_records: Final segmented page records, not yet OCR'd (mutated: segments added and reordered,
                ``glyph_height`` stored).
            work_directory: Conversion output directory.
            stage_progress: Receives the step's progress.

        Returns:
            Number of added segments.

        Raises:
            WorkerFailedError: If the layout worker fails.
            ValueError: If a page image cannot be read.
        """
        page_measurements = [self._measure_page(page_record) for page_record in page_records]
        font_statistics = collect_font_statistics(page_records)
        if font_statistics is None:
            stage_progress.advance_stage(1, 1, "too few paragraphs for residual text statistics")
            return 0
        blocks_by_page = page_residual_blocks(page_records, page_measurements, font_statistics)
        proposals_by_page = self._propose_boxes(page_records, blocks_by_page, work_directory, stage_progress)
        added_segments = {page_record["page"]: self._add_page_segments(page_record, blocks_by_page[page_record["page"]],
                                                                     proposals_by_page.get(page_record["page"], []))
                          for page_record in page_records if page_record["page"] in blocks_by_page}
        write_residual_report(work_directory / RESIDUAL_DIRECTORY, added_segments)
        added_count = sum(len(block_segments) for block_segments in added_segments.values())
        stage_progress.advance_stage(1, 1, f"{added_count} missed blocks added on {len(added_segments)} pages")
        return added_count

    def _measure_page(self, page_record: dict) -> PageGlyphs:
        """Measure one page and store its segment glyph heights.

        Args:
            page_record: Page record (mutated: ``glyph_height`` stored).

        Returns:
            The measurements.
        """
        page_glyphs = self._glyph_meter.measure_page(page_record)
        store_glyph_heights(page_record, page_glyphs)
        return page_glyphs

    def _propose_boxes(self, page_records: list[dict], blocks_by_page: dict[int, list[ResidualBlock]],
                       work_directory: Path, stage_progress: StageProgress) -> dict[int, list[dict]]:
        """Segment again the pages with missed paragraphs.

        Args:
            page_records: Final segmented page records.
            blocks_by_page: Residual blocks per page number.
            work_directory: Conversion output directory.
            stage_progress: Receives the layout progress.

        Returns:
            Mapping page number -> proposed boxes, for pages with missed paragraphs only.
        """
        rescanned_pages = {page_record["page"]: Path(page_record["image"]) for page_record in page_records
                           if any(residual_block.block_label == TEXT_LABEL
                                  for residual_block in blocks_by_page.get(page_record["page"], []))}
        if not rescanned_pages:
            return {}
        residual_directory = work_directory / RESIDUAL_DIRECTORY
        staged_images = stage_cleaned_pages(rescanned_pages, residual_directory)
        page_proposals = self._box_proposer.propose_boxes(staged_images, residual_directory, stage_progress)
        return dict(zip(sorted(rescanned_pages), page_proposals, strict=True))

    @staticmethod
    def _add_page_segments(page_record: dict, residual_blocks: list[ResidualBlock],
                           proposed_boxes: list[dict]) -> BlockSegments:
        """Build the segments of a page's residual blocks and insert them in reading order.

        Args:
            page_record: Page record (mutated: ``segments`` replaced by the merged, renumbered list; each block
                sees the segments added before it, so two blocks never take the same proposal).
            residual_blocks: Blocks of the page.
            proposed_boxes: Layout proposals of the page (empty when it was not segmented again).

        Returns:
            (block, segment) pairs of the added segments.
        """
        block_segments = []
        for residual_block in residual_blocks:
            new_segment = residual_segment(residual_block, proposed_boxes, page_record)
            page_record["segments"] = insert_by_position(page_record["segments"], [new_segment])
            block_segments.append((residual_block, new_segment))
        return block_segments
