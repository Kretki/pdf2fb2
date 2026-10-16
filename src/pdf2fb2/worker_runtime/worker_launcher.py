"""Starts model workers as child processes in their own virtual environments."""

import os

# B404/S404: subprocess is used only with a fixed argv list and no shell.
import subprocess  # nosec B404  # noqa: S404
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TextIO

from pdf2fb2.worker_runtime.worker_environment import WorkerEnvironment
from worker_protocol.progress_events import parse_progress_line

# Variables of the orchestrator's venv that must not leak into a worker venv.
ISOLATION_VARIABLES = ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT")
LOG_SUFFIX = ".log"
LOG_TAIL_LINES = 15

ProgressEventHandler = Callable[[dict], None]


class WorkerFailedError(RuntimeError):
    """Raised when a model worker exits with an unexpected code or returns an unusable result."""


@dataclass(frozen=True)
class WorkerRun:
    """Outcome of one worker process.

    Attributes:
        exit_code: Process exit code.
        log_file: File with everything the worker printed except progress events.
    """

    exit_code: int
    log_file: Path


class WorkerLauncher(Protocol):
    """Runs a worker on a job file.

    Usage:
        Injected into the layout and OCR stages; ``SubprocessWorkerLauncher`` is the
        production implementation, tests may pass a fake.
    """

    def launch_worker(self, worker_environment: WorkerEnvironment, job_file: Path,
                      event_handler: ProgressEventHandler) -> WorkerRun:
        """Run the worker to completion.

        Args:
            worker_environment: Which worker to start.
            job_file: Absolute path of the job file.
            event_handler: Called with every progress event the worker emits.

        Returns:
            Exit code and log file of the run.
        """
        ...


def build_worker_variables() -> dict[str, str]:
    """Copy the current environment without the orchestrator's venv variables.

    Returns:
        Environment mapping for the child process (unbuffered output enabled).
    """
    worker_variables = {variable_name: variable_value for variable_name, variable_value in os.environ.items()
                        if variable_name not in ISOLATION_VARIABLES}
    worker_variables["PYTHONUNBUFFERED"] = "1"
    return worker_variables


def forward_worker_output(output_lines: Iterable[bytes], log_stream: TextIO,
                          event_handler: ProgressEventHandler) -> None:
    """Send progress events to the handler and every other line to the log.

    Args:
        output_lines: Raw lines printed by the worker (stdout and stderr merged), UTF-8.
        log_stream: Open text log file.
        event_handler: Receives decoded progress events.

    Returns:
        None.
    """
    for raw_line in output_lines:
        output_line = raw_line.decode("utf-8", errors="replace")
        progress_event = parse_progress_line(output_line)
        if progress_event is None:
            log_stream.write(output_line)
        else:
            event_handler(progress_event)


def read_log_tail(log_file: Path, line_count: int = LOG_TAIL_LINES) -> str:
    """Return the last lines of a worker log for error messages.

    Args:
        log_file: Worker log file.
        line_count: Number of lines to keep, > 0.

    Returns:
        The last lines joined with newlines, or an empty string if the log is unreadable.
    """
    try:
        log_lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(log_lines[-line_count:])


def describe_worker_failure(worker_title: str, worker_run: WorkerRun) -> str:
    """Build a failure message that includes the end of the worker log.

    Args:
        worker_title: Human-readable worker name, e.g. ``Layout worker``.
        worker_run: Outcome of the failed run.

    Returns:
        Multi-line message for ``WorkerFailedError``.
    """
    log_tail = read_log_tail(worker_run.log_file)
    return (f"{worker_title} failed with exit code {worker_run.exit_code}. Full log: {worker_run.log_file}"
            + (f"\n--- last log lines ---\n{log_tail}" if log_tail else ""))


class SubprocessWorkerLauncher:
    """Starts ``<worker venv python> -m <module> --job-file <path>`` and streams its output.

    Usage:
        Create once and pass to the stages. Worker output goes to ``<job file>.log``;
        progress events go to the given handler.
    """

    def launch_worker(self, worker_environment: WorkerEnvironment, job_file: Path,
                      event_handler: ProgressEventHandler) -> WorkerRun:
        """Run the worker to completion.

        Args:
            worker_environment: Which worker to start.
            job_file: Absolute path of the job file.
            event_handler: Called with every progress event the worker emits.

        Returns:
            Exit code and log file of the run.

        Raises:
            FileNotFoundError: If the worker venv has not been created.
        """
        command_arguments = [str(worker_environment.interpreter_path()), "-m", worker_environment.module_name,
                             "--job-file", str(job_file.resolve())]
        log_file = job_file.with_suffix(LOG_SUFFIX)
        with log_file.open("w", encoding="utf-8") as log_stream:
            # B603/S603: argv list with the worker venv interpreter, no shell.
            with subprocess.Popen(  # nosec B603  # noqa: S603
                    command_arguments, cwd=worker_environment.project_directory, env=build_worker_variables(),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as worker_process:
                forward_worker_output(worker_process.stdout, log_stream, event_handler)
        return WorkerRun(exit_code=worker_process.returncode, log_file=log_file)
