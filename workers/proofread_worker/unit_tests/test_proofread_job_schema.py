"""Round-trip and validation tests of the proofreading job contract."""

from pathlib import Path

import pytest

from worker_protocol.proofread_job_schema import ProofreadJob, ProofreadSettings, TextBlock


def build_job(work_directory: Path) -> ProofreadJob:
    """A valid job inside ``work_directory``.

    Args:
        work_directory: Existing directory.

    Returns:
        ProofreadJob.
    """
    return ProofreadJob(work_directory=work_directory, model_directory=work_directory, device_name="cpu",
                        text_blocks=(TextBlock("p", ("One", "two.")), TextBlock("heading", ("Title",))),
                        block_checks=(1,), break_checks=(1,),
                        proofread_settings=ProofreadSettings(-9.0, 1200, 12, 100), result_file="result.json")


def test_payload_round_trip(tmp_path: Path) -> None:
    """Serialising and decoding gives the same job."""
    proofread_job = build_job(tmp_path)
    assert ProofreadJob.from_payload(proofread_job.to_payload()) == proofread_job


def test_out_of_range_check_is_rejected(tmp_path: Path) -> None:
    """Check indexes must refer to blocks of the job."""
    job_payload = build_job(tmp_path).to_payload()
    job_payload["block_checks"] = [5]
    with pytest.raises(ValueError):
        ProofreadJob.from_payload(job_payload)


def test_blank_words_are_rejected(tmp_path: Path) -> None:
    """Words must be non-blank strings."""
    job_payload = build_job(tmp_path).to_payload()
    job_payload["blocks"][0]["words"] = ["ok", "  "]
    with pytest.raises(ValueError):
        ProofreadJob.from_payload(job_payload)


def test_result_path_traversal_is_rejected(tmp_path: Path) -> None:
    """The result must stay inside the working directory."""
    job_payload = build_job(tmp_path).to_payload()
    job_payload["result_file"] = "../result.json"
    with pytest.raises(ValueError):
        ProofreadJob.from_payload(job_payload)
