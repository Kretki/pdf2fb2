"""Convenience launcher: ``uv run main.py book.pdf -o out`` (same as ``uv run pdf2fb2``).

All logic lives in the ``pdf2fb2`` package; the model workers run in their own
venvs under ``workers/`` (see README.md).
"""

import sys

from pdf2fb2.interface.command_line import run_command_line

if __name__ == "__main__":
    sys.exit(run_command_line())
