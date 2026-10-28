"""Judge entry of the proofread worker venv (vision-language check of uncertain verifier units).

Usage:
    workers/proofread_worker/.venv/bin/python -m proofread_worker.judge_entry --job-file <work_dir>/judge_job.json

Exit code 0 on success, 2 for an invalid job file; model errors exit non-zero.
"""

import argparse
import sys
from pathlib import Path

from worker_protocol.json_files import read_json_document, write_json_document
from worker_protocol.progress_events import JUDGE_PROGRESS_EVENT, MODEL_LOADING_EVENT, emit_progress_event

INVALID_JOB_EXIT_CODE = 2


def run_judge_file(judge_file: Path) -> int:
    """Validate and run a judge job.

    Args:
        judge_file: Job written by the orchestrator.

    Returns:
        Exit code: 0 on success, 2 if the job file is invalid.
    """
    from proofread_worker.judge_execution import run_judge_job
    from proofread_worker.visual_judge import load_judge
    from worker_protocol.judge_job_schema import JudgeJob

    try:
        judge_job = JudgeJob.from_payload(read_json_document(judge_file))
    except (OSError, ValueError) as job_error:
        print(f"Invalid judge job: {job_error}", file=sys.stderr)
        return INVALID_JOB_EXIT_CODE
    emit_progress_event(MODEL_LOADING_EVENT)
    result_document = run_judge_job(judge_job, load_judge(judge_job.model_directory, judge_job.device_name),
                                    lambda done_count, total_count: emit_progress_event(
                                        JUDGE_PROGRESS_EVENT, done=done_count, total=total_count))
    write_json_document(judge_job.resolve_inside(judge_job.result_file), result_document)
    return 0


def run_judge_worker(argument_list: list[str] | None = None) -> int:
    """Parse ``--job-file`` and run the judge job.

    Usage:
        Entry point of ``python -m proofread_worker.judge_entry`` (started by the orchestrator).

    Args:
        argument_list: Arguments without the program name; None means ``sys.argv[1:]``.

    Returns:
        Exit code.
    """
    argument_parser = argparse.ArgumentParser(description="Judge uncertain verifier units against line images.")
    argument_parser.add_argument("--job-file", dest="job_file", type=Path, required=True)
    return run_judge_file(argument_parser.parse_args(argument_list).job_file)


if __name__ == "__main__":
    sys.exit(run_judge_worker())
