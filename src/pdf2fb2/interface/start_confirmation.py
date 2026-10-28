"""Asks the admin to start (Enter) or abort (Esc) a run after its time estimate is shown.

Only an interactive terminal is asked; with redirected input the run starts at once. Keys are read in cbreak
mode, so a single Enter or Esc answers; escape sequences of other keys (arrows) are ignored.
"""

import os
import select
import sys
import termios
import tty
from typing import TextIO

ENTER_KEYS = frozenset({"\r", "\n"})
ESCAPE_KEY = "\x1b"
INTERRUPT_KEY = "\x03"
SEQUENCE_WAIT_SECONDS = 0.05
PROMPT_TEXT = "Press Enter to start, Esc to abort."


def read_key(input_descriptor: int) -> str:
    """Read one byte from the terminal, unbuffered (a buffered read would hide the rest of a sequence).

    Args:
        input_descriptor: File descriptor of the terminal in cbreak mode.

    Returns:
        The byte as a one-character string, or ``""`` at the end of input.
    """
    return os.read(input_descriptor, 1).decode("latin-1")


def has_pending_input(input_descriptor: int) -> bool:
    """Tell whether more key bytes are waiting (the rest of an escape sequence).

    Args:
        input_descriptor: File descriptor of the terminal in cbreak mode.

    Returns:
        True when input arrives within ``SEQUENCE_WAIT_SECONDS``.
    """
    readable_streams, _, _ = select.select([input_descriptor], [], [], SEQUENCE_WAIT_SECONDS)
    return bool(readable_streams)


def drain_sequence(input_descriptor: int) -> None:
    """Discard the rest of an escape sequence.

    Args:
        input_descriptor: File descriptor of the terminal in cbreak mode.

    Returns:
        None.
    """
    while has_pending_input(input_descriptor):
        read_key(input_descriptor)


def wait_for_answer(input_descriptor: int) -> bool:
    """Read keys until Enter or a lone Esc.

    Args:
        input_descriptor: File descriptor of the terminal in cbreak mode.

    Returns:
        True for Enter, False for Esc, Ctrl-C or the end of input.
    """
    while True:
        pressed_key = read_key(input_descriptor)
        if pressed_key in ENTER_KEYS:
            return True
        if pressed_key in ("", INTERRUPT_KEY):
            return False
        if pressed_key == ESCAPE_KEY:
            if not has_pending_input(input_descriptor):
                return False
            drain_sequence(input_descriptor)


def confirm_start(estimate_lines: list[str], input_stream: TextIO = sys.stdin,
                  output_stream: TextIO = sys.stderr) -> bool:
    """Show the estimate and wait for the admin's answer.

    Usage:
        Called by the command line and the benchmark before converting.

    Args:
        estimate_lines: Lines of the time estimate.
        input_stream: Keyboard input.
        output_stream: Where the estimate and the prompt are printed.

    Returns:
        True to start the run, False to abort it. Non-interactive input always starts.
    """
    for estimate_line in estimate_lines:
        print(estimate_line, file=output_stream)
    if not input_stream.isatty():
        return True
    print(PROMPT_TEXT, file=output_stream, flush=True)
    input_descriptor = input_stream.fileno()
    saved_settings = termios.tcgetattr(input_descriptor)
    try:
        tty.setcbreak(input_descriptor)
        return wait_for_answer(input_descriptor)
    except KeyboardInterrupt:
        return False
    finally:
        termios.tcsetattr(input_descriptor, termios.TCSADRAIN, saved_settings)
