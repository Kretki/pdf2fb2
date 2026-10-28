"""Second OCR read of every verified segment from a slightly larger crop (verifier evidence).

A different crop gives the OCR model a different view of the same print; where the two reads disagree a unit is
less certain. The text is stored in ``segment["second_read_text"]`` (and so in ``segments.json``, where the
verifier export finds it); segments that already have one are not read again.
"""

from pathlib import Path

from pdf2fb2.pipeline.stage_contracts import SegmentRecognizer
from pdf2fb2.progress.progress_contracts import StageProgress

SECOND_READ_DIRECTORY = "second_read"
MARGIN_RATIO = 0.04
MARGIN_PIXELS = 6


def enlarged_box(segment_box: list[float], page_size: list[int]) -> list[int]:
    """Segment box grown by a margin, clipped to the page.

    Args:
        segment_box: ``[x1, y1, x2, y2]`` in page pixels.
        page_size: ``[width, height]``.

    Returns:
        Enlarged integer box.
    """
    margin_size = int((segment_box[3] - segment_box[1]) * MARGIN_RATIO) + MARGIN_PIXELS
    return [max(0, int(segment_box[0]) - margin_size), max(0, int(segment_box[1]) - margin_size),
            min(int(page_size[0]), int(segment_box[2]) + margin_size),
            min(int(page_size[1]), int(segment_box[3]) + margin_size)]


def second_read_records(target_segments: list[tuple[dict, dict]]) -> list[dict]:
    """Pseudo page records with one enlarged ``text`` segment per target.

    Args:
        target_segments: ``(page record, segment)`` pairs to read again.

    Returns:
        Page records in the shape the OCR stage reads.
    """
    records_by_page: dict[int, dict] = {}
    for page_record, page_segment in target_segments:
        pseudo_record = records_by_page.setdefault(page_record["page"], {
            "page": page_record["page"], "image": page_record["image"], "size": page_record["size"], "segments": []})
        pseudo_record["segments"].append({"bbox": enlarged_box(page_segment["bbox"], page_record["size"]),
                                          "label": "text", "order": len(pseudo_record["segments"]), "score": 1.0})
    return list(records_by_page.values())


def read_segments_again(region_reader: SegmentRecognizer, target_segments: list[tuple[dict, dict]],
                        work_directory: Path, stage_progress: StageProgress) -> None:
    """Run the second read and store its texts on the segments.

    Usage:
        Called by ``VerificationFeedback`` when the second read is enabled.

    Args:
        region_reader: OCR stage.
        target_segments: ``(page record, segment)`` pairs (segments mutated).
        work_directory: Book output directory (the read uses its own sub-directory and cache).
        stage_progress: Progress sink.

    Returns:
        None.
    """
    pending_segments = [(page_record, page_segment) for page_record, page_segment in target_segments
                        if "second_read_text" not in page_segment]
    if not pending_segments:
        return
    pseudo_records = second_read_records(pending_segments)
    read_directory = work_directory / SECOND_READ_DIRECTORY
    read_directory.mkdir(parents=True, exist_ok=True)
    region_reader.recognize_segments(pseudo_records, read_directory, stage_progress)
    read_segments = [page_segment for pseudo_record in pseudo_records for page_segment in pseudo_record["segments"]]
    records_order = {page_record["page"]: record_index for record_index, page_record in enumerate(pseudo_records)}
    # Stable sort: the same grouping (pages in record order, targets in order within a page) as ``read_segments``.
    ordered_targets = sorted(pending_segments, key=lambda target_pair: records_order[target_pair[0]["page"]])
    for (_, page_segment), read_segment in zip(ordered_targets, read_segments, strict=True):
        page_segment["second_read_text"] = read_segment.get("text") or ""
