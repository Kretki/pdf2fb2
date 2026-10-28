"""Turns per-stage progress into one overall progress value and a time-left estimate.

The overall fraction is time-based: elapsed / (elapsed + estimated time left), where the time left comes from the
stored speeds of this machine (see ``timekeeping``). A slow final stage (proofreading) therefore keeps a long estimate
instead of looking almost done.
"""

import time
from collections.abc import Callable

from pdf2fb2.progress.progress_contracts import ProgressDisplay, RunTimer
from pdf2fb2.progress.stage_names import (  # noqa: F401 - re-exported for the stages
    ASSEMBLING_STAGE,
    CLEANING_STAGE,
    DRAWING_STAGE,
    HANDWRITING_STAGE,
    PICTURE_STAGE,
    PROOFREADING_STAGE,
    RECOGNITION_STAGE,
    RENDERING_STAGE,
    RESCAN_SEGMENTATION_STAGE,
    RESIDUAL_STAGE,
    REVIEW_STAGE,
    SEGMENT_REVIEW_STAGE,
    SEGMENTATION_STAGE,
    STYLING_STAGE,
    VERIFICATION_STAGE,
)


class PipelineProgress:
    """Keeps the overall bar monotonic and forwards the time-left estimate to the display.

    Usage:
        Create per conversion with the active stages and a run timer, hand ``stage_tracker(name)`` to each stage,
        report workload sizes with ``revise_workload`` and call ``close_progress`` at the end.
    """

    def __init__(self, progress_display: ProgressDisplay, stage_names: list[str], run_timer: RunTimer,
                 clock_function: Callable[[], float] = time.monotonic) -> None:
        """Bind the display and the timer.

        Args:
            progress_display: Display that renders the overall progress.
            stage_names: Active stages in execution order.
            run_timer: Estimates the time left from the stage reports.
            clock_function: Monotonic clock in seconds (injectable for tests).
        """
        self._progress_display = progress_display
        self._stage_names = frozenset(stage_names)
        self._run_timer = run_timer
        self._clock_function = clock_function
        self._started_at = clock_function()
        self._shown_fraction = 0.0

    def stage_tracker(self, stage_name: str) -> "TimedStageProgress":
        """Return the progress sink of one stage.

        Args:
            stage_name: Active stage name given to the constructor.

        Returns:
            Tracker that reports the stage's units to the timer.

        Raises:
            KeyError: If the stage is not active.
        """
        if stage_name not in self._stage_names:
            raise KeyError(stage_name)
        return TimedStageProgress(self, stage_name)

    def revise_workload(self, **workload_sizes: int) -> None:
        """Forward workload sizes that became known to the timer.

        Args:
            **workload_sizes: ``page_count``, ``text_page_count``, ``token_count`` or ``inspected_token_count``.

        Returns:
            None.
        """
        self._run_timer.revise_workload(**workload_sizes)

    def show_stage_fraction(self, stage_name: str, stage_fraction: float, status_text: str) -> None:
        """Record a stage report and redraw the overall progress, never moving the bar backwards.

        Args:
            stage_name: Reporting stage.
            stage_fraction: Its completed share, 0..1.
            status_text: Current step description.

        Returns:
            None.
        """
        self._run_timer.record_report(stage_name, stage_fraction)
        remaining_seconds = self._run_timer.remaining_seconds()
        elapsed_seconds = self._clock_function() - self._started_at
        run_seconds = elapsed_seconds + remaining_seconds
        overall_fraction = elapsed_seconds / run_seconds if run_seconds > 0 else 0.0
        self._shown_fraction = max(self._shown_fraction, min(overall_fraction, 1.0))
        self._progress_display.show_progress(self._shown_fraction, status_text, remaining_seconds)

    def show_message(self, message_text: str) -> None:
        """Forward a persistent message to the display.

        Args:
            message_text: One line of text.

        Returns:
            None.
        """
        self._progress_display.show_message(message_text)

    def close_progress(self, finished: bool) -> None:
        """Store the measured speeds, fill the bar (on success) and close the display.

        Args:
            finished: True when the conversion completed, False after an error.

        Returns:
            None.
        """
        try:
            self._run_timer.learn_speeds()
        except OSError as metrics_error:
            self.show_message(f"pdf2fb2: speed metrics not saved: {metrics_error}")
        if finished:
            self._shown_fraction = 1.0
            self._progress_display.show_progress(1.0, "done", 0.0)
        self._progress_display.close_display()


class TimedStageProgress:
    """``StageProgress`` of one stage: reports its share to the pipeline progress.

    Usage:
        Obtained from ``PipelineProgress.stage_tracker``.
    """

    def __init__(self, pipeline_progress: PipelineProgress, stage_name: str) -> None:
        """Bind the tracker to its stage.

        Args:
            pipeline_progress: Owner that renders the overall value.
            stage_name: Stage this tracker reports for.
        """
        self._pipeline_progress = pipeline_progress
        self._stage_name = stage_name

    def advance_stage(self, completed_units: int, total_units: int, status_text: str) -> None:
        """Report stage progress.

        Args:
            completed_units: Finished units (clamped to 0..total_units).
            total_units: Units in the stage (0 is treated as 1).
            status_text: Short description of the current step.

        Returns:
            None.
        """
        unit_count = max(total_units, 1)
        stage_fraction = min(max(completed_units, 0), unit_count) / unit_count
        self._pipeline_progress.show_stage_fraction(self._stage_name, stage_fraction, status_text)

    def report_message(self, message_text: str) -> None:
        """Show a persistent message.

        Args:
            message_text: One line of text.

        Returns:
            None.
        """
        self._pipeline_progress.show_message(message_text)
