"""``verify_report.json``: what the token verifier flagged in every segment and what the feedback loop changed."""

import json
from pathlib import Path

VERIFY_REPORT_FILE = "verify_report.json"


def write_verify_report(work_directory: Path, cases_by_key: dict, accepted_count: int) -> dict:
    """Write the report and return its summary.

    Args:
        work_directory: Output directory.
        cases_by_key: ``SegmentCase`` per segment key (after the loop).
        accepted_count: Segments whose text changed.

    Returns:
        Summary counts (segments, flagged units, deleted units, re-read regions, changed segments).
    """
    segment_entries = {segment_key: {**segment_case.decision_notes,
                                     "regions": [list(read_region.region_box)
                                                 for read_region in segment_case.read_regions]}
                       for segment_key, segment_case in cases_by_key.items()
                       if segment_case.decision_notes.get("misread") or segment_case.decision_notes.get("inserted")}
    verify_summary = {
        "verified_segments": len(cases_by_key),
        "misread_units": sum(len(segment_entry["misread"]) for segment_entry in segment_entries.values()),
        "inserted_units": sum(len(segment_entry["inserted"]) for segment_entry in segment_entries.values()),
        "deleted_units": sum(len(segment_entry["deleted"]) for segment_entry in segment_entries.values()),
        "reread_regions": sum(len(segment_entry["regions"]) for segment_entry in segment_entries.values()),
        "changed_segments": accepted_count}
    (work_directory / VERIFY_REPORT_FILE).write_text(json.dumps(
        {"summary": verify_summary, "segments": segment_entries}, ensure_ascii=False, indent=1), encoding="utf-8")
    return verify_summary
