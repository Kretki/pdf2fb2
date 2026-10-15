"""Command-line entry of the layout worker.

Usage:
    workers/layout_worker/.venv/bin/python -m layout_worker --job-file <work_dir>/layout_job.json

Reads a ``LayoutJob`` (see ``worker_protocol.job_schema``), segments every page
and writes the result document named in the job. Exit code 0 on success,
2 for an invalid job file.
"""

import argparse
import sys
from pathlib import Path

from layout_worker.job_execution import run_layout_job
from layout_worker.paddle_detector import PaddleLayoutDetector
from worker_protocol.job_schema import LayoutJob
from worker_protocol.json_files import read_json_document, write_json_document
from worker_protocol.progress_events import MODEL_LOADING_EVENT, emit_progress_event

INVALID_JOB_EXIT_CODE = 2


def parse_arguments(argument_list: list[str] | None) -> argparse.Namespace:
    """Parse the worker command line.

    Usage:
        Called by ``run_worker``.

    Args:
        argument_list: Arguments without the program name; None means ``sys.argv[1:]``.

    Returns:
        Namespace with ``job_file`` (Path).
    """
    argument_parser = argparse.ArgumentParser(description="Segment page images with PP-DocLayoutV3.")
    argument_parser.add_argument("--job-file", dest="job_file", type=Path, required=True)
    return argument_parser.parse_args(argument_list)


def run_worker(argument_list: list[str] | None = None) -> int:
    """Load the job, run the layout model and write the result.

    Usage:
        Entry point for ``python -m layout_worker`` and the ``layout-worker`` script.

    Args:
        argument_list: Arguments without the program name; None means ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 on success, 2 if the job file is invalid.
        Model errors propagate as exceptions (non-zero exit).
    """
    parsed_arguments = parse_arguments(argument_list)
    try:
        layout_job = LayoutJob.from_payload(read_json_document(parsed_arguments.job_file))
    except (OSError, ValueError) as job_error:
        print(f"layout_worker: invalid job file: {job_error}", file=sys.stderr)
        return INVALID_JOB_EXIT_CODE
    emit_progress_event(MODEL_LOADING_EVENT)
    layout_detector = PaddleLayoutDetector(layout_job.model_directory, layout_job.device_name)
    result_document = run_layout_job(layout_job, layout_detector)
    write_json_document(layout_job.resolve_inside(layout_job.result_file), result_document)
    return 0


if __name__ == "__main__":
    sys.exit(run_worker())
