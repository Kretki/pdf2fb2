"""Marks job contract: round trip, masked regions and traversal rejection."""

from pathlib import Path

import pytest

from worker_protocol.marks_job_schema import MarksJob


def build_job(work_directory: Path, page_regions: tuple) -> MarksJob:
    """A one-page job.

    Args:
        work_directory: Existing directory used as work and checkpoint directory.
        page_regions: Masked regions of the page(s).

    Returns:
        The job.
    """
    return MarksJob(work_directory, work_directory, "gpu:0", ("pages/page_0001.png",), "marks",
                    "marks_result.json", False, page_regions)


def test_round_trip_with_regions(tmp_path: Path) -> None:
    """A job with picture regions survives serialisation."""
    marks_job = build_job(tmp_path, (((10.0, 20.0, 300.0, 400.0),),))
    assert MarksJob.from_payload(marks_job.to_payload()) == marks_job


def test_missing_regions_default_to_none(tmp_path: Path) -> None:
    """Old payloads without ``masked_regions`` decode to one empty tuple per page."""
    job_payload = build_job(tmp_path, ((),)).to_payload()
    del job_payload["masked_regions"]
    assert MarksJob.from_payload(job_payload).page_regions() == ((),)


def test_invalid_inputs_are_rejected(tmp_path: Path) -> None:
    """Traversal paths, malformed boxes and wrong region counts raise ValueError."""
    job_payload = build_job(tmp_path, ((),)).to_payload()
    for hostile_payload in ({**job_payload, "page_images": ["../secret.png"]},
                            {**job_payload, "masked_regions": [[[5, 5, 1, 1]]]},
                            {**job_payload, "masked_regions": [[[0, 0, 1, "x"]]]},
                            {**job_payload, "masked_regions": [[], []]}):
        with pytest.raises(ValueError):
            MarksJob.from_payload(hostile_payload)
