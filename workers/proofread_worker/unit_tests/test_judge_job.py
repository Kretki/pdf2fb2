"""Unit tests of the judge job: schema validation and the job runner (the model is replaced by a stub)."""

from PIL import Image

from proofread_worker import judge_execution
from worker_protocol.judge_job_schema import JudgeItem, JudgeJob


def test_judge_job_round_trip_and_path_check(tmp_path) -> None:
    """A judge job survives serialisation; an image path outside the work directory is rejected."""
    judge_job = JudgeJob(tmp_path, tmp_path, "cpu", (JudgeItem("1:0#2", "judge_lines/a.png", "the rnodern age",
                                                               "rnodern"),), "judge_job_result.json")
    assert JudgeJob.from_payload(judge_job.to_payload()) == judge_job
    bad_payload = {**judge_job.to_payload(), "items": [{**judge_job.judge_items[0].to_payload(),
                                                        "image_file": "../outside.png"}]}
    try:
        JudgeJob.from_payload(bad_payload)
    except ValueError:
        return
    raise AssertionError("path traversal accepted")


def test_judge_runner_reports_every_item(tmp_path, monkeypatch) -> None:
    """Every item gets a rounded error probability in job order; progress reaches the total."""
    (tmp_path / "judge_lines").mkdir()
    Image.new("RGB", (40, 10), "white").save(tmp_path / "judge_lines" / "a.png")
    judge_items = (JudgeItem("1:0#1", "judge_lines/a.png", "the rnodern age", "rnodern"),
                   JudgeItem("1:0#2", "judge_lines/a.png", "the rnodern age", "age"))
    judge_job = JudgeJob(tmp_path, tmp_path, "cpu", judge_items, "judge_job_result.json")
    monkeypatch.setattr(judge_execution, "error_probability",
                        lambda _, line_image, line_text, target_text: 0.912345 if target_text == "rnodern" else 0.1)
    progress_calls: list[tuple[int, int]] = []
    result_document = judge_execution.run_judge_job(judge_job, None, lambda done_count, total_count:
                                                    progress_calls.append((done_count, total_count)))
    assert result_document == {"items": [{"key": "1:0#1", "error_probability": 0.9123},
                                         {"key": "1:0#2", "error_probability": 0.1}]}
    assert progress_calls[-1] == (2, 2)
