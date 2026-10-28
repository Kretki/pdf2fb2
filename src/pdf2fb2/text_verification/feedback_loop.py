"""Feedback between the token verifier and the OCR: re-read what the verifier flags, keep what reads better.

1. Every book-text segment is split into units and labelled (correct / misread / inserted) by the verifier.
2. Units flagged ``inserted`` that have no detected word box under them are deleted: the verifier and the page
   agree that the text is not printed (a hallucinated continuation, a duplicated line).
3. The text lines of ``misread`` units (and of ``inserted`` units that do have ink, e.g. a reader's note over the
   text) are cropped tightly and read again by the OCR model: a line crop gives the model far less context to
   "complete" from than the whole paragraph.
4. Segments with a re-read are labelled again; the new text is kept only when its mean error probability is
   lower by ``ACCEPT_MARGIN``. Every decision goes to ``verify_report.json``.

Without word boxes (``--no-styles``) only step 1 runs and the flags are reported.

The first pass sends every unit with its page evidence (``unit_evidence``): neighbouring segments as context, OCR
confidence, word boxes, an optional second OCR read (``second_read``) and optional word crops. Units the verifier
is unsure about can be judged by a vision-language model against their line image (``uncertain_units``). The
recheck labels the original and the candidate text alike (context only), so their error masses are comparable.
"""

from dataclasses import dataclass, field
from pathlib import Path

from pdf2fb2.pipeline.stage_contracts import SegmentRecognizer
from pdf2fb2.progress.progress_contracts import StageProgress
from pdf2fb2.segments.label_roles import is_book_text
from pdf2fb2.text_verification.evidence_items import CANDIDATE_SUFFIX, ORIGINAL_SUFFIX, evidence_items, recheck_items
from pdf2fb2.text_verification.region_mapping import (
    ReadRegion,
    UnitPlacement,
    place_units,
    reread_regions,
    unprinted_units,
)
from pdf2fb2.text_verification.second_read import read_segments_again
from pdf2fb2.text_verification.temporary_images import remove_temporary_images
from pdf2fb2.text_verification.text_splicing import TextEdit, apply_edits, deletion_edit, flat_reading
from pdf2fb2.text_verification.uncertain_units import UnitJudge, judged_probabilities
from pdf2fb2.text_verification.unit_flags import error_mass, flag_units
from pdf2fb2.text_verification.verification_contracts import FeedbackOptions, ItemLabeler
from pdf2fb2.text_verification.verify_report import write_verify_report
from pdf2fb2.text_verification.verify_units import VerifyUnit, verify_units

ACCEPT_MARGIN = 0.02
REREAD_DIRECTORY = "verify_rereads"


@dataclass
class SegmentCase:
    """Verification state of one segment.

    Attributes:
        page_record: Page of the segment.
        page_segment: The segment (its ``text`` is replaced when a change is accepted).
        segment_units: Units of the original text.
        text_edits: Deletions decided in the first pass.
        read_regions: Regions to re-read.
        decision_notes: Report entries.
        unit_placement: Placement on the word boxes (set when the evidence is built).
        context_before: Units of the page before the segment.
        context_after: Units of the page after the segment.
    """

    page_record: dict
    page_segment: dict
    segment_units: list[VerifyUnit]
    text_edits: list[TextEdit] = field(default_factory=list)
    read_regions: list[ReadRegion] = field(default_factory=list)
    decision_notes: dict = field(default_factory=dict)
    unit_placement: UnitPlacement | None = None
    context_before: tuple[str, ...] = ()
    context_after: tuple[str, ...] = ()


def segment_cases(page_records: list[dict]) -> dict[str, SegmentCase]:
    """Book-text segments with text, keyed ``<page>:<segment index>``.

    Args:
        page_records: OCR'd page records.

    Returns:
        Cases in reading order.
    """
    return {f"{page_record['page']}:{segment_index}": SegmentCase(page_record, page_segment,
                                                                   verify_units(page_segment["text"]))
            for page_record in page_records for segment_index, page_segment in enumerate(page_record["segments"])
            if is_book_text(page_segment) and (page_segment.get("text") or "").strip()}


def plan_segment(segment_case: SegmentCase, probability_rows: list[list[float]],
                 class_thresholds: dict[str, float]) -> None:
    """Decide deletions and re-reads of one segment from the first labels.

    Args:
        segment_case: Segment state (mutated).
        probability_rows: First-pass probabilities.
        class_thresholds: Decision thresholds.

    Returns:
        None.
    """
    unit_flags = flag_units(probability_rows, class_thresholds)
    unit_placement = segment_case.unit_placement or place_units(segment_case.segment_units,
                                                                segment_case.page_segment.get("words") or [])
    deleted_units = unprinted_units(unit_placement, list(unit_flags.inserted_units),
                                    segment_case.segment_units)
    segment_case.text_edits = [deletion_edit(segment_case.page_segment["text"], segment_case.segment_units[unit_index])
                               for unit_index in deleted_units]
    reread_targets = sorted(set(unit_flags.misread_units) | (set(unit_flags.inserted_units) - set(deleted_units)))
    if unit_placement is not None and reread_targets:
        segment_case.read_regions = reread_regions(unit_placement, reread_targets,
                                                   tuple(segment_case.page_record["size"]))
    segment_case.decision_notes = {
        "misread": [segment_case.segment_units[unit_index].unit_text for unit_index in unit_flags.misread_units],
        "inserted": [segment_case.segment_units[unit_index].unit_text for unit_index in unit_flags.inserted_units],
        "deleted": [segment_case.segment_units[unit_index].unit_text for unit_index in deleted_units],
        "error_mass": round(error_mass(probability_rows, frozenset(deleted_units)), 4)}


def region_records(cases_by_key: dict[str, SegmentCase]) -> tuple[list[dict], list[tuple[str, int]]]:
    """Pseudo page records holding one ``text`` segment per region to re-read.

    Args:
        cases_by_key: Planned segments.

    Returns:
        ``(page records, (segment key, region index) per region segment in record order)``.
    """
    records_by_page: dict[int, dict] = {}
    region_owners: list[tuple[str, int]] = []
    for segment_key, segment_case in cases_by_key.items():
        for region_index, read_region in enumerate(segment_case.read_regions):
            page_record = segment_case.page_record
            pseudo_record = records_by_page.setdefault(page_record["page"], {
                "page": page_record["page"], "image": page_record["image"], "size": page_record["size"],
                "segments": []})
            pseudo_record["segments"].append({"bbox": list(read_region.region_box), "label": "text",
                                              "order": len(pseudo_record["segments"]), "score": 1.0})
            region_owners.append((segment_key, region_index))
    return list(records_by_page.values()), region_owners


def candidate_text(segment_case: SegmentCase, region_texts: dict[int, str]) -> str:
    """Segment text with the deletions and the re-read regions applied.

    Args:
        segment_case: Planned segment.
        region_texts: Region index -> re-read text.

    Returns:
        New text.
    """
    region_edits = [TextEdit(segment_case.segment_units[read_region.first_unit].start_offset,
                             segment_case.segment_units[read_region.last_unit].end_offset,
                             flat_reading(region_texts.get(region_index, "")))
                    for region_index, read_region in enumerate(segment_case.read_regions)
                    if region_texts.get(region_index, "").strip()]
    outside_deletions = [text_edit for text_edit in segment_case.text_edits
                         if not any(region_edit.start_offset <= text_edit.start_offset < region_edit.end_offset
                                    for region_edit in region_edits)]
    return apply_edits(segment_case.page_segment["text"], region_edits + outside_deletions)


class VerificationFeedback:
    """Runs the verify -> re-read -> verify loop over the OCR'd pages.

    Usage:
        Build with the verification stage and the OCR stage and call ``verify_segments`` after the word
        boxes are detected and before word styles are applied.
    """

    def __init__(self, item_labeler: ItemLabeler, region_reader: SegmentRecognizer,
                 feedback_options: FeedbackOptions | None = None, unit_judge: UnitJudge | None = None) -> None:
        """Store the stages.

        Args:
            item_labeler: Verifier stage.
            region_reader: OCR stage used to re-read regions (and for the second read).
            feedback_options: Optional evidence; None uses the defaults (no crops, no second read).
            unit_judge: Vision-language judge of uncertain units, or None.
        """
        self._item_labeler = item_labeler
        self._region_reader = region_reader
        self._feedback_options = feedback_options or FeedbackOptions()
        self._unit_judge = unit_judge

    def verify_segments(self, page_records: list[dict], work_directory: Path, stage_progress: StageProgress) -> dict:
        """Verify all segments, re-read flagged lines and keep the better texts; remove the temporary crops.

        Args:
            page_records: OCR'd page records (segment ``text`` replaced when a change is accepted; the
                previous text is kept in ``pre_verify_text``).
            work_directory: Output directory (``verify_crops/``, ``judge_lines/`` and ``second_read/ocr_crops/``
                are deleted when verification ends, also after a failure).
            stage_progress: Progress sink of the verification stage.

        Returns:
            Summary counts.
        """
        try:
            return self._verify_pages(page_records, work_directory, stage_progress)
        finally:
            remove_temporary_images(work_directory)

    def _verify_pages(self, page_records: list[dict], work_directory: Path, stage_progress: StageProgress) -> dict:
        """Run the verification loop over all segments.

        Args:
            page_records: OCR'd page records (mutated as in ``verify_segments``).
            work_directory: Output directory.
            stage_progress: Progress sink.

        Returns:
            Summary counts.
        """
        cases_by_key = segment_cases(page_records)
        if self._feedback_options.second_read:
            read_segments_again(self._region_reader, [(segment_case.page_record, segment_case.page_segment)
                                                      for segment_case in cases_by_key.values()],
                                work_directory, stage_progress)
        first_outcome = self._item_labeler.label_items(
            evidence_items(cases_by_key, work_directory, self._feedback_options.write_crops), work_directory,
            "verify_job", stage_progress)
        unit_probabilities = first_outcome.unit_probabilities if self._unit_judge is None else judged_probabilities(
            cases_by_key, first_outcome.unit_probabilities, self._unit_judge, self._feedback_options.judge_band,
            work_directory, stage_progress)
        for segment_key, segment_case in cases_by_key.items():
            plan_segment(segment_case, unit_probabilities.get(segment_key, []), first_outcome.class_thresholds)
        candidate_texts = self._reread_candidates(cases_by_key, work_directory, stage_progress)
        accepted_count = self._accept_better(cases_by_key, candidate_texts, work_directory, stage_progress)
        return write_verify_report(work_directory, cases_by_key, accepted_count)

    def _reread_candidates(self, cases_by_key: dict[str, SegmentCase], work_directory: Path,
                           stage_progress: StageProgress) -> dict[str, str]:
        """Re-read the planned regions and build the candidate texts.

        Args:
            cases_by_key: Planned segments.
            work_directory: Output directory.
            stage_progress: Progress sink.

        Returns:
            Segment key -> candidate text for segments with any edit.
        """
        pseudo_records, region_owners = region_records(cases_by_key)
        region_texts: dict[str, dict[int, str]] = {}
        if pseudo_records:
            reread_directory = work_directory / REREAD_DIRECTORY
            reread_directory.mkdir(parents=True, exist_ok=True)
            self._region_reader.recognize_segments(pseudo_records, reread_directory, stage_progress)
            region_segments = [page_segment for pseudo_record in pseudo_records
                               for page_segment in pseudo_record["segments"]]
            for (segment_key, region_index), region_segment in zip(region_owners, region_segments, strict=True):
                region_texts.setdefault(segment_key, {})[region_index] = region_segment.get("text") or ""
        return {segment_key: candidate_text(segment_case, region_texts.get(segment_key, {}))
                for segment_key, segment_case in cases_by_key.items()
                if segment_case.text_edits or segment_key in region_texts}

    def _accept_better(self, cases_by_key: dict[str, SegmentCase], candidate_texts: dict[str, str],
                       work_directory: Path, stage_progress: StageProgress) -> int:
        """Label the candidates and keep those that read better.

        Args:
            cases_by_key: Planned segments.
            candidate_texts: Segment key -> candidate text.
            work_directory: Output directory.
            stage_progress: Progress sink.

        Returns:
            Number of segments whose text changed.
        """
        changed_texts = {segment_key: new_text for segment_key, new_text in candidate_texts.items()
                         if new_text and new_text != cases_by_key[segment_key].page_segment["text"]}
        second_outcome = self._item_labeler.label_items(
            recheck_items(cases_by_key, changed_texts), work_directory, "verify_recheck_job", stage_progress)
        accepted_count = 0
        for segment_key, new_text in changed_texts.items():
            segment_case = cases_by_key[segment_key]
            old_mass = error_mass(second_outcome.unit_probabilities.get(f"{segment_key}{ORIGINAL_SUFFIX}", []))
            new_mass = error_mass(second_outcome.unit_probabilities.get(f"{segment_key}{CANDIDATE_SUFFIX}", []))
            segment_case.decision_notes.update({"candidate": new_text, "candidate_error_mass": round(new_mass, 4),
                                                "recheck_error_mass": round(old_mass, 4)})
            if new_mass + ACCEPT_MARGIN <= old_mass:
                segment_case.page_segment["pre_verify_text"] = segment_case.page_segment["text"]
                segment_case.page_segment["text"] = new_text
                segment_case.decision_notes["accepted"] = True
                accepted_count += 1
        return accepted_count
