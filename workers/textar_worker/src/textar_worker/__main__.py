"""pdf2fb2 worker entry: word-level bold / italic / underline / strikeout of text segments.

Usage:
    workers/textar_worker/.venv/bin/python -m textar_worker --job-file <work_dir>/style_job.json

Reads a ``StyleJob`` (see ``worker_protocol.style_job_schema``), writes the result document named
in the job. Exit code 0 on success, 2 for an invalid job file; model errors exit non-zero.
"""

import argparse
import sys
from pathlib import Path

from textar_worker.job_execution import load_style_models, run_style_job
from worker_protocol.json_files import read_json_document, write_json_document
from worker_protocol.progress_events import MODEL_LOADING_EVENT, STYLE_PAGE_EVENT, emit_progress_event
from worker_protocol.style_job_schema import StyleJob

INVALID_JOB_EXIT_CODE = 2


def parse_arguments(argument_list: list[str] | None) -> argparse.Namespace:
    """Parse the worker command line.

    Args:
        argument_list: Arguments without the program name; None means ``sys.argv[1:]``.

    Returns:
        Namespace with ``job_file`` (Path).
    """
    argument_parser = argparse.ArgumentParser(description="Detect word text attributes with TexTAR.")
    argument_parser.add_argument("--job-file", dest="job_file", type=Path, required=True)
    return argument_parser.parse_args(argument_list)


def run_worker(argument_list: list[str] | None = None) -> int:
    """Validate the job file, run it and write the result.

    Usage:
        Entry point of ``python -m textar_worker``.

    Args:
        argument_list: Arguments without the program name; None means ``sys.argv[1:]``.

    Returns:
        Exit code: 0 on success, 2 if the job file is invalid.
    """
    parsed_arguments = parse_arguments(argument_list)
    try:
        style_job = StyleJob.from_payload(read_json_document(parsed_arguments.job_file))
    except (OSError, ValueError) as job_error:
        print(f"Invalid style job: {job_error}", file=sys.stderr)
        return INVALID_JOB_EXIT_CODE
    emit_progress_event(MODEL_LOADING_EVENT)
    style_models = load_style_models(style_job)
    result_document = run_style_job(
        style_job, style_models,
        lambda page_number, page_count: emit_progress_event(STYLE_PAGE_EVENT, page=page_number, total=page_count))
    write_json_document(style_job.resolve_inside(style_job.result_file), result_document)
    return 0


if __name__ == "__main__":
    sys.exit(run_worker())
