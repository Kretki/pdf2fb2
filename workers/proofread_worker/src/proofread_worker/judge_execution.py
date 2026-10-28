"""Runs a ``JudgeJob``: one error probability per uncertain unit."""

from collections.abc import Callable

from PIL import Image

from proofread_worker.visual_judge import LoadedJudge, error_probability
from worker_protocol.judge_job_schema import JudgeJob

PROBABILITY_DIGITS = 4
JudgeProgress = Callable[[int, int], None]


def run_judge_job(judge_job: JudgeJob, loaded_judge: LoadedJudge, progress_callback: JudgeProgress) -> dict:
    """Judge every item of the job.

    Usage:
        Called by the worker entry point in ``--judge-job`` mode.

    Args:
        judge_job: Validated job.
        loaded_judge: Vision-language model.
        progress_callback: Receives ``(done items, total items)``.

    Returns:
        Result document (see ``worker_protocol.judge_job_schema``).

    Raises:
        ValueError: If an image path escapes the work directory.
    """
    result_items = []
    item_count = len(judge_job.judge_items)
    for item_number, judge_item in enumerate(judge_job.judge_items, start=1):
        with Image.open(judge_job.resolve_inside(judge_item.image_file)) as line_file:
            line_image = line_file.convert("RGB")
        result_items.append({"key": judge_item.item_key, "error_probability": round(
            error_probability(loaded_judge, line_image, judge_item.line_text, judge_item.target_text),
            PROBABILITY_DIGITS)})
        progress_callback(item_number, item_count)
    return {"items": result_items}
