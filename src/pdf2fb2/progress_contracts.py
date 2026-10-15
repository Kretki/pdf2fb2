"""Abstractions of progress reporting used by the pipeline and its stages."""

from typing import Protocol


class ProgressDisplay(Protocol):
    """Shows overall progress and a short status of the current step.

    Usage:
        Implemented by ``TerminalProgressDisplay`` (two-line live view),
        ``PlainProgressDisplay`` (log lines) and ``SilentProgressDisplay`` (tests).
    """

    def show_progress(self, overall_fraction: float, status_text: str) -> None:
        """Update the overall progress and the current status.

        Args:
            overall_fraction: Completed share of the whole conversion, 0..1.
            status_text: Short description of the current step.

        Returns:
            None.
        """
        ...

    def show_message(self, message_text: str) -> None:
        """Print a message that must stay visible (e.g. a warning) without breaking the display.

        Args:
            message_text: One line of text.

        Returns:
            None.
        """
        ...

    def close_display(self) -> None:
        """Finish the display; later console output starts on a clean line.

        Returns:
            None.
        """
        ...


class StageProgress(Protocol):
    """Progress sink of one pipeline stage.

    Usage:
        Passed to a stage; the stage reports its own units and never needs to know
        how its share maps to the overall bar.
    """

    def advance_stage(self, completed_units: int, total_units: int, status_text: str) -> None:
        """Report how much of this stage is done.

        Args:
            completed_units: Finished units, 0..total_units.
            total_units: Units in the stage (> 0; 0 is treated as 1).
            status_text: Short description such as ``segmenting page 7/40``.

        Returns:
            None.
        """
        ...

    def report_message(self, message_text: str) -> None:
        """Show a message that must stay visible after the run (e.g. a failed crop).

        Args:
            message_text: One line of text.

        Returns:
            None.
        """
        ...
