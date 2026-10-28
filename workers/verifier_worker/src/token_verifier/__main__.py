"""Entry point of the verifier worker: pipeline jobs and the admin commands (wiring only).

Usage:
    .venv/bin/python -m token_verifier --job-file <work_dir>/verify_job.json     # pipeline (started by pdf2fb2)
    .venv/bin/python -m token_verifier synth --split train                      # synthetic examples
    .venv/bin/python -m token_verifier train                                    # fine-tune base model
    .venv/bin/python -m token_verifier tune                                     # thresholds on valid
    .venv/bin/python -m token_verifier eval --split test                        # final report

Every admin command accepts ``--config settings.json`` (overrides ``default_settings.json``). Job mode exits
with 0 on success and 2 for an invalid job file.
"""

import argparse
import sys
from pathlib import Path

INVALID_JOB_EXIT_CODE = 2
SPLIT_CHOICES = ("train", "valid", "test")


def build_parser() -> argparse.ArgumentParser:
    """Command-line parser.

    Returns:
        Parser with ``--job-file`` and the admin subcommands.
    """
    argument_parser = argparse.ArgumentParser(prog="token_verifier", description="Token verifier worker.")
    argument_parser.add_argument("--job-file", dest="job_file", type=Path)
    subcommand_parsers = argument_parser.add_subparsers(dest="command_name")
    for command_name in ("synth", "train", "tune", "eval"):
        command_parser = subcommand_parsers.add_parser(command_name)
        command_parser.add_argument("--config", dest="config_path", type=Path, default=None)
        if command_name in ("synth", "eval"):
            command_parser.add_argument("--split", dest="split_name", choices=SPLIT_CHOICES,
                                        default="train" if command_name == "synth" else "test")
    return argument_parser


def run_job_file(job_file: Path) -> int:
    """Run one pipeline job.

    Args:
        job_file: Job written by the orchestrator.

    Returns:
        Exit code.
    """
    from token_verifier.inference.job_execution import run_verify_job
    from token_verifier.networks.model_loading import load_verifier
    from worker_protocol.json_files import read_json_document, write_json_document
    from worker_protocol.progress_events import MODEL_LOADING_EVENT, VERIFY_PROGRESS_EVENT, emit_progress_event
    from worker_protocol.verify_job_schema import VerifyJob

    try:
        verify_job = VerifyJob.from_payload(read_json_document(job_file))
    except (OSError, ValueError) as job_error:
        print(f"Invalid verify job: {job_error}", file=sys.stderr)
        return INVALID_JOB_EXIT_CODE
    emit_progress_event(MODEL_LOADING_EVENT)
    loaded_verifier = load_verifier(verify_job.model_directory, verify_job.device_name)
    result_document = run_verify_job(verify_job, loaded_verifier, lambda done_count, total_count: emit_progress_event(
        VERIFY_PROGRESS_EVENT, done=done_count, total=total_count))
    write_json_document(verify_job.resolve_inside(verify_job.result_file), result_document)
    return 0


def run_admin_command(parsed_arguments: argparse.Namespace) -> int:
    """Run an admin subcommand.

    Args:
        parsed_arguments: Parsed options.

    Returns:
        Exit code 0.
    """
    from token_verifier.datasets.synthesis_service import synthesize_split
    from token_verifier.evaluation.evaluation_service import run_evaluation, run_tuning
    from token_verifier.training.training_service import run_training
    from token_verifier.verifier_settings import load_settings

    verifier_settings = load_settings(parsed_arguments.config_path)
    command_handlers = {
        "synth": lambda: synthesize_split(verifier_settings.path_settings.data_directory, parsed_arguments.split_name,
                                          verifier_settings.synthesis_settings),
        "train": lambda: run_training(verifier_settings),
        "tune": lambda: run_tuning(verifier_settings),
        "eval": lambda: run_evaluation(verifier_settings, parsed_arguments.split_name),
    }
    print(command_handlers[parsed_arguments.command_name](), flush=True)
    return 0


def run_worker(argument_list: list[str] | None = None) -> int:
    """Dispatch to job mode or an admin command.

    Usage:
        Entry point of ``python -m token_verifier``.

    Args:
        argument_list: Arguments without the program name; None means ``sys.argv[1:]``.

    Returns:
        Exit code.
    """
    argument_parser = build_parser()
    parsed_arguments = argument_parser.parse_args(argument_list)
    if parsed_arguments.job_file is not None:
        return run_job_file(parsed_arguments.job_file)
    if parsed_arguments.command_name is None:
        argument_parser.error("give --job-file or a subcommand")
    return run_admin_command(parsed_arguments)


if __name__ == "__main__":
    sys.exit(run_worker())
