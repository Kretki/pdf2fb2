"""pdf2fb2 worker entry: language-model proofreading of the book text.

Usage:
    workers/proofread_worker/.venv/bin/python -m proofread_worker --job-file <work_dir>/proofread_job.json
    workers/proofread_worker/.venv/bin/python -m proofread_worker --judge-job <work_dir>/judge_job.json

Reads a ``ProofreadJob`` (see ``worker_protocol.proofread_job_schema``) or a ``JudgeJob`` (vision-language
check of uncertain verifier units, ``worker_protocol.judge_job_schema``) and writes the result document named in
the job. Exit code 0 on success, 2 for an invalid job file; model errors exit non-zero.
"""

import argparse
import sys
from pathlib import Path

from proofread_worker.job_execution import run_proofread_job
from proofread_worker.model_loading import load_language_model
from proofread_worker.token_scorer import TokenScorer
from worker_protocol.json_files import read_json_document, write_json_document
from worker_protocol.progress_events import MODEL_LOADING_EVENT, PROOFREAD_PROGRESS_EVENT, emit_progress_event
from worker_protocol.proofread_job_schema import ProofreadJob

INVALID_JOB_EXIT_CODE = 2


def parse_arguments(argument_list: list[str] | None) -> argparse.Namespace:
    """Parse the worker command line.

    Args:
        argument_list: Arguments without the program name; None means ``sys.argv[1:]``.

    Returns:
        Namespace with ``job_file`` or ``judge_job`` (Path; exactly one is set).
    """
    argument_parser = argparse.ArgumentParser(description="Score book text with a local language model.")
    job_group = argument_parser.add_mutually_exclusive_group(required=True)
    job_group.add_argument("--job-file", dest="job_file", type=Path)
    job_group.add_argument("--judge-job", dest="judge_job", type=Path)
    return argument_parser.parse_args(argument_list)


def report_progress(phase_name: str, done_count: int, total_count: int) -> None:
    """Emit one progress event.

    Args:
        phase_name: ``scoring`` or ``checking``.
        done_count: Finished units.
        total_count: All units of the phase.

    Returns:
        None.
    """
    emit_progress_event(PROOFREAD_PROGRESS_EVENT, phase=phase_name, done=done_count, total=total_count)


def run_worker(argument_list: list[str] | None = None) -> int:
    """Validate the job file, run it and write the result.

    Usage:
        Entry point of ``python -m proofread_worker``.

    Args:
        argument_list: Arguments without the program name; None means ``sys.argv[1:]``.

    Returns:
        Exit code: 0 on success, 2 if the job file is invalid.
    """
    parsed_arguments = parse_arguments(argument_list)
    if parsed_arguments.judge_job is not None:
        from proofread_worker.judge_entry import run_judge_file

        return run_judge_file(parsed_arguments.judge_job)
    try:
        proofread_job = ProofreadJob.from_payload(read_json_document(parsed_arguments.job_file))
    except (OSError, ValueError) as job_error:
        print(f"Invalid proofreading job: {job_error}", file=sys.stderr)
        return INVALID_JOB_EXIT_CODE
    emit_progress_event(MODEL_LOADING_EVENT)
    token_scorer = TokenScorer(load_language_model(proofread_job.model_directory, proofread_job.device_name))
    result_document = run_proofread_job(proofread_job, token_scorer, report_progress)
    write_json_document(proofread_job.resolve_inside(proofread_job.result_file), result_document)
    return 0


if __name__ == "__main__":
    sys.exit(run_worker())
