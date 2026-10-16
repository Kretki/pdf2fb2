"""pdf2fb2 worker entry: clean reader marks from rendered pages before layout and OCR.

Usage:
    workers/readermarks_worker/.venv/bin/python -m readermarks.job_worker --job-file <work_dir>/marks_job.json

Reads a ``MarksJob`` (see ``worker_protocol.marks_job_schema``), cleans every page and writes the
result document named in the job. Exit code 0 on success, 2 for an invalid job file.
"""

import argparse
import sys
from pathlib import Path

import torch

from readermarks.inference.cleaning_service import clean_page_batch
from readermarks.inference.model_loading import load_models
from readermarks.settings.model_settings import ReaderMarkSettings, load_settings
from worker_protocol.json_files import read_json_document, write_json_document
from worker_protocol.marks_job_schema import MarksJob
from worker_protocol.path_safety import relative_to_directory
from worker_protocol.progress_events import MARKS_PAGE_EVENT, MODEL_LOADING_EVENT, emit_progress_event

INVALID_JOB_EXIT_CODE = 2


def parse_arguments(argument_list: list[str] | None) -> argparse.Namespace:
    """Parse the worker command line.

    Args:
        argument_list: Arguments without the program name; None means ``sys.argv[1:]``.

    Returns:
        Namespace with ``job_file`` (Path).
    """
    argument_parser = argparse.ArgumentParser(description="Remove reader marks from page images.")
    argument_parser.add_argument("--job-file", dest="job_file", type=Path, required=True)
    return argument_parser.parse_args(argument_list)


def torch_device(device_name: str) -> torch.device:
    """Map a pdf2fb2 device name (``gpu:0``, ``cuda:0``, ``cpu``) to a torch device.

    Args:
        device_name: Validated device name.

    Returns:
        Torch device; CUDA names fall back to CPU when CUDA is unavailable.
    """
    if device_name == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name.replace("gpu", "cuda"))


def job_settings(marks_job: MarksJob) -> ReaderMarkSettings:
    """Default settings with the checkpoint directory taken from the job.

    Args:
        marks_job: Validated job.

    Returns:
        Settings.
    """
    settings = load_settings()
    return settings.model_copy(
        update={
            "paths": settings.path_settings.model_copy(update={"checkpoint_directory": marks_job.checkpoint_directory})
        }
    )


def run_marks_job(marks_job: MarksJob) -> dict:
    """Load the models and clean every page of the job (masked picture regions blanked first).

    Args:
        marks_job: Validated job.

    Returns:
        Result document (see the module docstring of ``marks_job_schema``).
    """
    emit_progress_event(MODEL_LOADING_EVENT)
    loaded_models = load_models(job_settings(marks_job), torch_device(marks_job.device_name))
    input_paths = [marks_job.resolve_inside(page_image) for page_image in marks_job.page_images]
    written_pages = clean_page_batch(
        loaded_models.page_cleaner,
        input_paths,
        marks_job.resolve_inside(marks_job.output_directory),
        marks_job.write_debug,
        lambda page_number, page_count: emit_progress_event(MARKS_PAGE_EVENT, page=page_number, total=page_count),
        marks_job.page_regions(),
    )
    return {
        "pages": [
            {
                "image_file": page_image,
                "clean_file": relative_to_directory(marks_job.work_directory, written_page.clean_path),
                "marks_file": relative_to_directory(marks_job.work_directory, written_page.marks_path),
                "has_marks": written_page.has_marks,
                "mark_count": written_page.mark_count,
            }
            for page_image, written_page in zip(marks_job.page_images, written_pages, strict=True)
        ]
    }


def run_worker(argument_list: list[str] | None = None) -> int:
    """Validate the job file, run it and write the result.

    Usage:
        Entry point of ``python -m readermarks.job_worker``.

    Args:
        argument_list: Arguments without the program name; None means ``sys.argv[1:]``.

    Returns:
        Exit code: 0 on success, 2 if the job file is invalid. Model errors propagate (non-zero exit).
    """
    parsed_arguments = parse_arguments(argument_list)
    try:
        marks_job = MarksJob.from_payload(read_json_document(parsed_arguments.job_file))
    except (OSError, ValueError) as job_error:
        print(f"readermarks worker: invalid job file: {job_error}", file=sys.stderr)
        return INVALID_JOB_EXIT_CODE
    write_json_document(marks_job.resolve_inside(marks_job.result_file), run_marks_job(marks_job))
    return 0


if __name__ == "__main__":
    sys.exit(run_worker())
