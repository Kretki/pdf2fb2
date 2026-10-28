"""Measures how long each stage runs, from the progress reports of the stages.

A stage becomes active at its first report and stays active until another stage reports; a stage may become
active several times (assembling runs before and after proofreading), its time is summed. A stage is finished
once it reports all its units, or once a stage later in the plan becomes active after it (assembling starting
before proofreading does not finish proofreading).
"""

from collections.abc import Callable
from dataclasses import dataclass

from pdf2fb2.timekeeping.time_estimation import StageState


@dataclass
class StageRecord:
    """Mutable timing of one stage.

    Attributes:
        active_seconds: Summed active time, not counting the current activation.
        startup_seconds: Active time before the first completed unit (None: no unit done yet).
        stage_fraction: Last reported completed share, 0..1 (a stage may start a second pass from 0).
        activation_number: Sequence number of its latest activation (later activations have larger numbers).
    """

    active_seconds: float = 0.0
    startup_seconds: float | None = None
    stage_fraction: float = 0.0
    activation_number: int = 0


class StageClock:
    """Times the stages of one conversion.

    Usage:
        Created by ``PipelineProgress``; ``record_report`` is called for every stage report.
    """

    def __init__(self, stage_names: tuple[str, ...], clock_function: Callable[[], float]) -> None:
        """Prepare empty records.

        Args:
            stage_names: Active stages in plan order.
            clock_function: Monotonic clock in seconds.
        """
        self._plan_positions = {stage_name: stage_position for stage_position, stage_name in enumerate(stage_names)}
        self._clock_function = clock_function
        self._stage_records: dict[str, StageRecord] = {}
        self._active_stage: str | None = None
        self._activated_at = clock_function()
        self._activation_count = 0

    def record_report(self, stage_name: str, stage_fraction: float) -> None:
        """Take one progress report of a stage.

        Args:
            stage_name: Reporting stage.
            stage_fraction: Its completed share, 0..1.

        Returns:
            None.
        """
        if stage_name != self._active_stage:
            self._switch_stage(stage_name)
        stage_record = self._stage_records[stage_name]
        stage_record.stage_fraction = stage_fraction
        if stage_fraction > 0.0 and stage_record.startup_seconds is None:
            stage_record.startup_seconds = self._active_seconds(stage_name)

    def stop_clock(self) -> None:
        """End the current activation (at the end of the run).

        Returns:
            None.
        """
        self._switch_stage(None)

    def stage_states(self) -> dict[str, StageState]:
        """State of every started stage, for the remaining-time estimate.

        Returns:
            State per stage name.
        """
        return {stage_name: StageState(self._active_seconds(stage_name), stage_record.stage_fraction,
                                       self._is_finished(stage_name))
                for stage_name, stage_record in self._stage_records.items()}

    def finished_records(self) -> dict[str, StageRecord]:
        """Timing of every finished stage, for learning the speeds.

        Returns:
            Record per finished stage name (active time includes the current activation).
        """
        return {stage_name: StageRecord(self._active_seconds(stage_name),
                                        stage_record.startup_seconds or 0.0, stage_record.stage_fraction)
                for stage_name, stage_record in self._stage_records.items() if self._is_finished(stage_name)}

    def _switch_stage(self, stage_name: str | None) -> None:
        """Close the current activation and open the next one.

        Args:
            stage_name: Stage becoming active (None: none).

        Returns:
            None.
        """
        current_time = self._clock_function()
        if self._active_stage is not None:
            self._stage_records[self._active_stage].active_seconds += current_time - self._activated_at
        self._active_stage = stage_name
        self._activated_at = current_time
        if stage_name is not None:
            self._activation_count += 1
            self._stage_records.setdefault(stage_name, StageRecord()).activation_number = self._activation_count

    def _active_seconds(self, stage_name: str) -> float:
        """Summed active time of a stage, including the running activation.

        Args:
            stage_name: Started stage.

        Returns:
            Seconds.
        """
        running_seconds = self._clock_function() - self._activated_at if stage_name == self._active_stage else 0.0
        return self._stage_records[stage_name].active_seconds + running_seconds

    def _is_finished(self, stage_name: str) -> bool:
        """Tell whether a started stage is over.

        Args:
            stage_name: Started stage.

        Returns:
            True when it reported all its units or a stage later in the plan became active after it.
        """
        stage_record = self._stage_records[stage_name]
        plan_position = self._plan_positions.get(stage_name, -1)
        return stage_record.stage_fraction >= 1.0 or any(
            self._plan_positions.get(other_name, -1) > plan_position
            and other_record.activation_number > stage_record.activation_number
            for other_name, other_record in self._stage_records.items())
