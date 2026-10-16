"""Round-trip and validation tests of the style job contract."""

from pathlib import Path

import pytest

from worker_protocol.style_job_schema import StyleJob, StylePage


def build_job(work_directory: Path) -> StyleJob:
    """A valid two-page job inside ``work_directory``.

    Args:
        work_directory: Existing directory.

    Returns:
        StyleJob.
    """
    return StyleJob(work_directory=work_directory, textar_directory=work_directory, detector_directory=work_directory,
                    device_name="cpu", style_pages=(StylePage("pages/page_0001.png", ((1.0, 2.0, 30.0, 40.0),)),
                                                    StylePage("pages/page_0002.png", ())),
                    result_file="style_result.json")


def test_payload_round_trip(tmp_path: Path) -> None:
    """Serialising and decoding gives the same job."""
    style_job = build_job(tmp_path)
    assert StyleJob.from_payload(style_job.to_payload()) == style_job


def test_path_traversal_is_rejected(tmp_path: Path) -> None:
    """Image paths must stay inside the working directory."""
    job_payload = build_job(tmp_path).to_payload()
    job_payload["style_pages"][0]["image_file"] = "../outside.png"
    with pytest.raises(ValueError):
        StyleJob.from_payload(job_payload)


def test_invalid_region_is_rejected(tmp_path: Path) -> None:
    """Region boxes must be ordered."""
    job_payload = build_job(tmp_path).to_payload()
    job_payload["style_pages"][0]["text_regions"] = [[50, 50, 10, 10]]
    with pytest.raises(ValueError):
        StyleJob.from_payload(job_payload)
