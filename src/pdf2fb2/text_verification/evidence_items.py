"""Verifier job items of the segments of a book, with page context, page evidence and (optionally) word crops.

Item keys are the segment keys of ``feedback_loop`` (``<page>:<segment index>``).
"""

from pathlib import Path
from typing import Protocol

from PIL import Image

from pdf2fb2.text_verification.region_mapping import UnitPlacement, place_units
from pdf2fb2.text_verification.unit_evidence import page_contexts, unit_crops, unit_features
from pdf2fb2.text_verification.verify_units import VerifyUnit, verify_units
from worker_protocol.verify_job_schema import VerifyItem

ORIGINAL_SUFFIX = "|original"
CANDIDATE_SUFFIX = "|candidate"


class EvidenceCase(Protocol):
    """What the builder needs of a segment case (``feedback_loop.SegmentCase``).

    Attributes:
        page_record: Page record.
        page_segment: Segment record.
        segment_units: Units of the segment text.
        unit_placement: Placement on the word boxes (set by the builder).
        context_before: Units before the segment on its page (set by the builder).
        context_after: Units after the segment on its page (set by the builder).
    """

    page_record: dict
    page_segment: dict
    segment_units: list[VerifyUnit]
    unit_placement: UnitPlacement | None
    context_before: tuple[str, ...]
    context_after: tuple[str, ...]


def attach_page_context(cases_by_key: dict[str, EvidenceCase]) -> None:
    """Give every case the units of its neighbouring segments on the same page.

    Args:
        cases_by_key: Cases in reading order (mutated).

    Returns:
        None.
    """
    keys_by_page: dict[int, list[str]] = {}
    for segment_key, segment_case in cases_by_key.items():
        keys_by_page.setdefault(segment_case.page_record["page"], []).append(segment_key)
    for page_keys in keys_by_page.values():
        page_units = [[verify_unit.unit_text for verify_unit in cases_by_key[segment_key].segment_units]
                      for segment_key in page_keys]
        for segment_key, (context_before, context_after) in zip(page_keys, page_contexts(page_units), strict=True):
            cases_by_key[segment_key].context_before = context_before
            cases_by_key[segment_key].context_after = context_after


def case_crops(segment_key: str, segment_case: EvidenceCase, work_directory: Path,
               image_cache: dict[str, Image.Image]) -> tuple[str | None, ...]:
    """Word crops of one case (page images are opened once).

    Args:
        segment_key: Segment key.
        segment_case: Case with its placement.
        work_directory: Output directory (crop paths are relative to it).
        image_cache: Page image path -> opened image (mutated).

    Returns:
        Crop path or None per unit.
    """
    if segment_case.unit_placement is None:
        return ()
    image_path = str(segment_case.page_record["image"])
    if image_path not in image_cache:
        image_cache.clear()
        with Image.open(image_path) as page_file:
            image_cache[image_path] = page_file.convert("RGB")
    return unit_crops(image_cache[image_path], segment_case.unit_placement, segment_key.replace(":", "_"),
                      work_directory)


def evidence_items(cases_by_key: dict[str, EvidenceCase], work_directory: Path, write_crops: bool) -> list[VerifyItem]:
    """Items of all cases with context, features and optional crops.

    Usage:
        Called by ``VerificationFeedback`` before the first verification pass.

    Args:
        cases_by_key: Cases in reading order (placement and context are set on them).
        work_directory: Output directory of the job.
        write_crops: Write word crops for the image-and-text features.

    Returns:
        One item per case.
    """
    attach_page_context(cases_by_key)
    image_cache: dict[str, Image.Image] = {}
    verify_items = []
    for segment_key, segment_case in cases_by_key.items():
        segment_case.unit_placement = place_units(segment_case.segment_units,
                                                  segment_case.page_segment.get("words") or [])
        crop_paths = case_crops(segment_key, segment_case, work_directory, image_cache) if write_crops else ()
        verify_items.append(VerifyItem(segment_key, tuple(verify_unit.unit_text
                                                          for verify_unit in segment_case.segment_units),
                                       segment_case.context_before, segment_case.context_after,
                                       unit_features(segment_case.page_segment, segment_case.page_segment["text"],
                                                     segment_case.segment_units, segment_case.unit_placement),
                                       crop_paths))
    return verify_items


def recheck_items(cases_by_key: dict[str, EvidenceCase], changed_texts: dict[str, str]) -> list[VerifyItem]:
    """Recheck items: the original and the candidate text of every changed segment, both with page context only.

    Args:
        cases_by_key: Planned segments (with context).
        changed_texts: Segment key -> candidate text.

    Returns:
        Two items per changed segment (``|original`` and ``|candidate`` keys).
    """
    recheck_list = []
    for segment_key, new_text in changed_texts.items():
        segment_case = cases_by_key[segment_key]
        for key_suffix, item_text in ((ORIGINAL_SUFFIX, segment_case.page_segment["text"]),
                                      (CANDIDATE_SUFFIX, new_text)):
            recheck_list.append(VerifyItem(f"{segment_key}{key_suffix}", tuple(
                verify_unit.unit_text for verify_unit in verify_units(item_text)), segment_case.context_before,
                segment_case.context_after))
    return recheck_list
