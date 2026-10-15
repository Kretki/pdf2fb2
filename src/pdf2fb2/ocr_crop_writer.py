"""Saves crop images of the segments that still need OCR and describes them for progress."""

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from pdf2fb2.image_cropping import crop_segment_for_ocr
from pdf2fb2.label_roles import ocr_token_limit, requires_ocr
from pdf2fb2.ocr_text_cache import OcrTextCache, build_segment_key
from pdf2fb2.progress_contracts import StageProgress
from worker_protocol.ocr_job_schema import OcrCropItem
from worker_protocol.path_safety import relative_to_directory

OCR_CROPS_DIRECTORY = "ocr_crops"


@dataclass(frozen=True)
class PendingCrop:
    """A crop sent to the OCR worker plus its human-readable position.

    Attributes:
        ocr_item: Job item referencing the saved crop.
        block_label: Position such as ``block 2/5 on page 3/10``.
    """

    ocr_item: OcrCropItem
    block_label: str


def _ocr_segment_indexes(page_record: dict) -> list[int]:
    """List the indexes of the segments of a page that are read by OCR.

    Args:
        page_record: Page record with segments.

    Returns:
        Segment indexes in reading order.
    """
    return [segment_index for segment_index, page_segment in enumerate(page_record["segments"])
            if requires_ocr(page_segment["label"])]


class OcrCropWriter:
    """Writes the crops of uncached OCR segments into ``ocr_crops/``.

    Usage:
        Created by ``OcrStage`` for each run; call ``write_pending_crops``.
    """

    def __init__(self, work_directory: Path, text_cache: OcrTextCache, page_count: int) -> None:
        """Prepare the crops directory.

        Args:
            work_directory: Output directory; job paths are relative to it.
            text_cache: Cache of already read segments (those are skipped).
            page_count: Total number of pages, used in the block labels.
        """
        self._work_directory = work_directory
        self._crops_directory = work_directory / OCR_CROPS_DIRECTORY
        self._crops_directory.mkdir(parents=True, exist_ok=True)
        self._text_cache = text_cache
        self._page_count = page_count

    def write_pending_crops(self, page_records: list[dict], stage_progress: StageProgress) -> list[PendingCrop]:
        """Save a crop for every uncached OCR segment of every page.

        Args:
            page_records: Page records in page order.
            stage_progress: Receives ``cropping blocks on page k/n`` updates.

        Returns:
            Pending crops in job order.
        """
        pending_crops: list[PendingCrop] = []
        for page_position, page_record in enumerate(page_records, start=1):
            stage_progress.advance_stage(0, 1, f"cropping blocks on page {page_position}/{self._page_count}")
            pending_crops.extend(self._write_page_crops(page_record, page_position))
        return pending_crops

    def _write_page_crops(self, page_record: dict, page_position: int) -> list[PendingCrop]:
        """Save the crops of the uncached OCR segments of one page.

        Args:
            page_record: Page record.
            page_position: 1-based position of the page in the document.

        Returns:
            Pending crops of this page.
        """
        segment_indexes = _ocr_segment_indexes(page_record)
        uncached_blocks = [(block_number, segment_index)
                           for block_number, segment_index in enumerate(segment_indexes, start=1)
                           if build_segment_key(page_record["page"], page_record["segments"][segment_index])
                           not in self._text_cache]
        if not uncached_blocks:
            return []
        with Image.open(page_record["image"]) as page_file:
            page_image = page_file.convert("RGB")
        block_suffix = f"/{len(segment_indexes)} on page {page_position}/{self._page_count}"
        return [PendingCrop(ocr_item=self._save_crop(page_image, page_record, segment_index),
                            block_label=f"block {block_number}{block_suffix}")
                for block_number, segment_index in uncached_blocks]

    def _save_crop(self, page_image: Image.Image, page_record: dict, segment_index: int) -> OcrCropItem:
        """Save one segment crop and describe it as a job item.

        Args:
            page_image: RGB page image.
            page_record: Page record owning the segment.
            segment_index: Index of the segment in ``page_record["segments"]``.

        Returns:
            Job item referencing the saved crop.
        """
        page_segment = page_record["segments"][segment_index]
        crop_path = self._crops_directory / f"page_{page_record['page']:04d}_segment_{segment_index:03d}.png"
        crop_segment_for_ocr(page_image, page_segment["bbox"]).save(crop_path)
        return OcrCropItem(item_key=build_segment_key(page_record["page"], page_segment),
                           image_file=relative_to_directory(self._work_directory, crop_path),
                           max_new_tokens=ocr_token_limit(page_segment["label"]))
