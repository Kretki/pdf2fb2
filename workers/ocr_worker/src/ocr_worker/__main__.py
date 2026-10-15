"""Command-line entry of the OCR worker.

Usage:
    workers/ocr_worker/.venv/bin/python -m ocr_worker --job-file <work_dir>/ocr_job.json

Reads an ``OcrJobSpec`` (see ``worker_protocol.ocr_job_schema``), reads every crop with
Qianfan-OCR and appends one JSON line per crop to the job's result file.
Exit code 0 when all crops were read, 1 when some failed, 2 for an invalid job file.
"""

import argparse
import sys
from pathlib import Path

from ocr_worker.engine_factory import create_ocr_engine
from ocr_worker.job_execution import run_ocr_job
from worker_protocol.json_files import read_json_document
from worker_protocol.ocr_job_schema import OcrJobSpec
from worker_protocol.progress_events import MODEL_LOADING_EVENT, emit_progress_event

PARTIAL_FAILURE_EXIT_CODE = 1
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
    argument_parser = argparse.ArgumentParser(description="Read segment crops with Qianfan-OCR.")
    argument_parser.add_argument("--job-file", dest="job_file", type=Path, required=True)
    return argument_parser.parse_args(argument_list)


def run_worker(argument_list: list[str] | None = None) -> int:
    """Load the job, build the OCR engine and read every crop.

    Usage:
        Entry point for ``python -m ocr_worker`` and the ``ocr-worker`` script.

    Args:
        argument_list: Arguments without the program name; None means ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 all crops read, 1 some crops failed (see result file),
        2 invalid job file. Model loading errors propagate as exceptions.
    """
    parsed_arguments = parse_arguments(argument_list)
    try:
        ocr_request = OcrJobSpec.from_payload(read_json_document(parsed_arguments.job_file))
    except (OSError, ValueError) as job_error:
        print(f"ocr_worker: invalid job file: {job_error}", file=sys.stderr)
        return INVALID_JOB_EXIT_CODE
    emit_progress_event(MODEL_LOADING_EVENT)
    failure_count = run_ocr_job(ocr_request, create_ocr_engine(ocr_request))
    return PARTIAL_FAILURE_EXIT_CODE if failure_count else 0


if __name__ == "__main__":
    sys.exit(run_worker())
