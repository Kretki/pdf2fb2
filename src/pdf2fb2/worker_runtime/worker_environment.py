"""Locates the isolated virtual environment of each model worker."""

import os
from dataclasses import dataclass
from pathlib import Path

from pdf2fb2.project_paths import (
    LAYOUT_WORKER_DIRECTORY,
    OCR_WORKER_DIRECTORY,
    READERMARKS_WORKER_DIRECTORY,
    TEXTAR_WORKER_DIRECTORY,
)


@dataclass(frozen=True)
class WorkerEnvironment:
    """A worker project with its own venv, started as ``python -m <module_name>``.

    Attributes:
        project_directory: Worker project directory (contains pyproject.toml and .venv).
        module_name: Python package run with ``-m``.
        override_variable: Environment variable that may point to another interpreter.
    """

    project_directory: Path
    module_name: str
    override_variable: str

    def interpreter_path(self) -> Path:
        """Return the Python interpreter of the worker venv.

        Usage:
            Called by the launcher right before starting the worker.

        Returns:
            ``$<override_variable>`` if set, else ``<project_directory>/.venv/bin/python``.

        Raises:
            FileNotFoundError: If the interpreter does not exist (``uv sync`` not run yet).
        """
        override_value = os.environ.get(self.override_variable)
        interpreter = Path(override_value) if override_value else self.project_directory / ".venv" / "bin" / "python"
        if not interpreter.is_file():
            raise FileNotFoundError(
                f"Worker interpreter {interpreter} not found. Run `uv sync --project {self.project_directory}` "
                f"(or set {self.override_variable}).")
        return interpreter


LAYOUT_WORKER = WorkerEnvironment(LAYOUT_WORKER_DIRECTORY, "layout_worker", "PDF2FB2_LAYOUT_PYTHON")
OCR_WORKER = WorkerEnvironment(OCR_WORKER_DIRECTORY, "ocr_worker", "PDF2FB2_OCR_PYTHON")
MARKS_WORKER = WorkerEnvironment(READERMARKS_WORKER_DIRECTORY, "readermarks.job_worker", "PDF2FB2_MARKS_PYTHON")
STYLE_WORKER = WorkerEnvironment(TEXTAR_WORKER_DIRECTORY, "textar_worker", "PDF2FB2_STYLE_PYTHON")
