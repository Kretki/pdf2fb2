"""Console implementations of ``ProgressDisplay``.

``TerminalProgressDisplay`` keeps exactly two live lines (tqdm-style bar + current
step) using ANSI cursor control; ``PlainProgressDisplay`` is for redirected output.
"""

import shutil
import sys
import time
from collections.abc import Callable
from typing import TextIO

from pdf2fb2.progress_contracts import ProgressDisplay
from pdf2fb2.progress_formatting import compose_progress_line, fit_to_width

CURSOR_UP = "\x1b[1A"
CLEAR_LINE = "\x1b[2K"
REDRAW_INTERVAL_SECONDS = 0.1
PLAIN_REPORT_STEP = 0.1
FRACTION_TOLERANCE = 1e-9


def read_terminal_columns() -> int:
    """Return the current terminal width.

    Returns:
        Number of columns (80 when unknown).
    """
    return shutil.get_terminal_size(fallback=(80, 24)).columns


class TerminalProgressDisplay:
    """Two-line live view: the overall bar and, under it, the current step.

    Usage:
        Use when the output stream is an interactive terminal.
    """

    def __init__(self, title_text: str, output_stream: TextIO = sys.stderr,
                 clock_function: Callable[[], float] = time.monotonic,
                 columns_function: Callable[[], int] = read_terminal_columns) -> None:
        """Prepare the display; nothing is drawn until the first update.

        Args:
            title_text: Label in front of the bar (e.g. the PDF name).
            output_stream: Terminal stream to draw on.
            clock_function: Monotonic clock in seconds (injectable for tests).
            columns_function: Returns the terminal width (injectable for tests).
        """
        self._title_text = title_text
        self._output_stream = output_stream
        self._clock_function = clock_function
        self._columns_function = columns_function
        self._started_at = clock_function()
        self._last_drawn_at: float | None = None
        self._overall_fraction = 0.0
        self._status_text = "starting"
        self._lines_drawn = False

    def show_progress(self, overall_fraction: float, status_text: str) -> None:
        """Store the new state and redraw at most ten times per second.

        Args:
            overall_fraction: Completed share, 0..1.
            status_text: Current step description.

        Returns:
            None.
        """
        self._overall_fraction = overall_fraction
        self._status_text = status_text
        current_time = self._clock_function()
        if self._last_drawn_at is None or current_time - self._last_drawn_at >= REDRAW_INTERVAL_SECONDS:
            self._redraw_lines(current_time)

    def show_message(self, message_text: str) -> None:
        """Print a message above the two live lines and redraw them below it.

        Args:
            message_text: One line of text.

        Returns:
            None.
        """
        self._output_stream.write(self._return_to_first_line() + CLEAR_LINE + message_text + "\n")
        self._lines_drawn = False
        self._redraw_lines(self._clock_function())

    def close_display(self) -> None:
        """Draw the final bar, erase the status line and leave the cursor below the bar.

        Returns:
            None.
        """
        terminal_columns = self._columns_function()
        progress_line = compose_progress_line(self._title_text, self._overall_fraction,
                                              self._clock_function() - self._started_at, terminal_columns)
        self._output_stream.write(self._return_to_first_line() + CLEAR_LINE + progress_line + "\n" + CLEAR_LINE)
        self._output_stream.flush()
        self._lines_drawn = False

    def _return_to_first_line(self) -> str:
        """Control sequence that moves the cursor back to the bar line.

        Returns:
            ANSI sequence, or an empty string when nothing is drawn yet.
        """
        return "\r" + CURSOR_UP if self._lines_drawn else ""

    def _redraw_lines(self, current_time: float) -> None:
        """Overwrite the bar line and the status line in place.

        Args:
            current_time: Clock value used for the elapsed time.

        Returns:
            None. Writes to the output stream.
        """
        terminal_columns = self._columns_function()
        progress_line = compose_progress_line(self._title_text, self._overall_fraction,
                                              current_time - self._started_at, terminal_columns)
        status_line = fit_to_width("  " + self._status_text, terminal_columns)
        self._output_stream.write(self._return_to_first_line() + CLEAR_LINE + progress_line + "\n"
                                  + CLEAR_LINE + status_line)
        self._output_stream.flush()
        self._lines_drawn = True
        self._last_drawn_at = current_time


class PlainProgressDisplay:
    """Log-style progress for non-interactive output: one line per 10% step.

    Usage:
        Use when the output is redirected to a file or a pipe.
    """

    def __init__(self, output_stream: TextIO = sys.stderr) -> None:
        """Prepare the display.

        Args:
            output_stream: Stream that receives the lines.
        """
        self._output_stream = output_stream
        self._next_report_fraction = PLAIN_REPORT_STEP

    def show_progress(self, overall_fraction: float, status_text: str) -> None:
        """Print a line each time progress crosses the next 10% step.

        Args:
            overall_fraction: Completed share, 0..1.
            status_text: Current step description.

        Returns:
            None.
        """
        if overall_fraction + FRACTION_TOLERANCE < self._next_report_fraction:
            return
        print(f"{int(overall_fraction * 100):3d}% {status_text}", file=self._output_stream, flush=True)
        while self._next_report_fraction <= overall_fraction + FRACTION_TOLERANCE:
            self._next_report_fraction += PLAIN_REPORT_STEP

    def show_message(self, message_text: str) -> None:
        """Print a message line.

        Args:
            message_text: One line of text.

        Returns:
            None.
        """
        print(message_text, file=self._output_stream, flush=True)

    def close_display(self) -> None:
        """Nothing to clean up.

        Returns:
            None.
        """


class SilentProgressDisplay:
    """Display that shows nothing except messages.

    Usage:
        For tests and embedding the pipeline in other programs.
    """

    def show_progress(self, overall_fraction: float, status_text: str) -> None:
        """Ignore progress.

        Args:
            overall_fraction: Ignored.
            status_text: Ignored.

        Returns:
            None.
        """

    def show_message(self, message_text: str) -> None:
        """Print the message to stderr.

        Args:
            message_text: One line of text.

        Returns:
            None.
        """
        print(message_text, file=sys.stderr)

    def close_display(self) -> None:
        """Nothing to clean up.

        Returns:
            None.
        """


def create_progress_display(title_text: str, output_stream: TextIO = sys.stderr) -> ProgressDisplay:
    """Pick the live two-line display for terminals and the log-style display otherwise.

    Usage:
        Called by the command-line wiring.

    Args:
        title_text: Label in front of the bar.
        output_stream: Stream to draw on.

    Returns:
        A ``ProgressDisplay`` implementation.
    """
    if output_stream.isatty():
        return TerminalProgressDisplay(title_text, output_stream)
    return PlainProgressDisplay(output_stream)
