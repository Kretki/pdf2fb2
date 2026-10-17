"""Maps per-stage progress onto one overall progress value."""

from pdf2fb2.progress.progress_contracts import ProgressDisplay

RENDERING_STAGE = "rendering"
CLEANING_STAGE = "cleaning"
HANDWRITING_STAGE = "handwriting"
PICTURE_STAGE = "pictures"
SEGMENTATION_STAGE = "segmentation"
DRAWING_STAGE = "drawing"
RECOGNITION_STAGE = "recognition"
STYLING_STAGE = "styling"
PROOFREADING_STAGE = "proofreading"
RESCAN_SEGMENTATION_STAGE = "rescan_segmentation"
ASSEMBLING_STAGE = "assembling"

# Rough share of the run time of each stage; OCR dominates on real books.
STAGE_WEIGHTS = {RENDERING_STAGE: 6.0, SEGMENTATION_STAGE: 22.0, RECOGNITION_STAGE: 65.0, CLEANING_STAGE: 6.0,
                 RESCAN_SEGMENTATION_STAGE: 4.0, HANDWRITING_STAGE: 2.0, PICTURE_STAGE: 2.0, STYLING_STAGE: 4.0,
                 PROOFREADING_STAGE: 15.0, DRAWING_STAGE: 3.0, ASSEMBLING_STAGE: 4.0}


class PipelineProgress:
    """Splits the overall bar into weighted stage windows and keeps it monotonic.

    Usage:
        Create per conversion with the active stages, hand ``stage_tracker(name)``
        to each stage, and call ``close_progress`` at the end.
    """

    def __init__(self, progress_display: ProgressDisplay, stage_names: list[str]) -> None:
        """Compute the window of every active stage.

        Args:
            progress_display: Display that renders the overall progress.
            stage_names: Active stages in execution order (keys of ``STAGE_WEIGHTS``).

        Raises:
            KeyError: If a stage name is unknown.
        """
        self._progress_display = progress_display
        weight_total = sum(STAGE_WEIGHTS[stage_name] for stage_name in stage_names) or 1.0
        self._stage_windows: dict[str, tuple[float, float]] = {}
        window_start = 0.0
        for stage_name in stage_names:
            window_width = STAGE_WEIGHTS[stage_name] / weight_total
            self._stage_windows[stage_name] = (window_start, window_width)
            window_start += window_width
        self._shown_fraction = 0.0

    def stage_tracker(self, stage_name: str) -> "WeightedStageProgress":
        """Return the progress sink of one stage.

        Args:
            stage_name: Active stage name given to the constructor.

        Returns:
            Tracker that maps the stage's units into its window.

        Raises:
            KeyError: If the stage is not active.
        """
        window_start, window_width = self._stage_windows[stage_name]
        return WeightedStageProgress(self, window_start, window_width)

    def show_fraction(self, overall_fraction: float, status_text: str) -> None:
        """Forward progress to the display, never moving the bar backwards.

        Args:
            overall_fraction: Overall completed share, 0..1.
            status_text: Current step description.

        Returns:
            None.
        """
        self._shown_fraction = max(self._shown_fraction, min(overall_fraction, 1.0))
        self._progress_display.show_progress(self._shown_fraction, status_text)

    def show_message(self, message_text: str) -> None:
        """Forward a persistent message to the display.

        Args:
            message_text: One line of text.

        Returns:
            None.
        """
        self._progress_display.show_message(message_text)

    def close_progress(self, finished: bool) -> None:
        """Fill the bar (on success) and close the display.

        Args:
            finished: True when the conversion completed, False after an error.

        Returns:
            None.
        """
        if finished:
            self.show_fraction(1.0, "done")
        self._progress_display.close_display()


class WeightedStageProgress:
    """``StageProgress`` that maps stage units into the stage's window of the overall bar.

    Usage:
        Obtained from ``PipelineProgress.stage_tracker``.
    """

    def __init__(self, pipeline_progress: PipelineProgress, window_start: float, window_width: float) -> None:
        """Bind the tracker to its window.

        Args:
            pipeline_progress: Owner that renders the overall value.
            window_start: Overall fraction where the stage starts, 0..1.
            window_width: Overall fraction covered by the stage, 0..1.
        """
        self._pipeline_progress = pipeline_progress
        self._window_start = window_start
        self._window_width = window_width

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
        self._pipeline_progress.show_fraction(self._window_start + self._window_width * stage_fraction, status_text)

    def report_message(self, message_text: str) -> None:
        """Show a persistent message.

        Args:
            message_text: One line of text.

        Returns:
            None.
        """
        self._pipeline_progress.show_message(message_text)
