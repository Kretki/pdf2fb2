"""Timing of one conversion: remaining-time estimate while it runs, speed learning when it ends."""

import time
from collections.abc import Callable
from dataclasses import replace

from pdf2fb2.timekeeping.default_speeds import default_stage_speeds
from pdf2fb2.timekeeping.speed_metrics import StageSample, TextSample, learn_run
from pdf2fb2.timekeeping.speed_metrics_store import SpeedMetricsStore
from pdf2fb2.timekeeping.stage_clock import StageClock, StageRecord
from pdf2fb2.timekeeping.time_estimation import ConversionPlan, remaining_run_seconds
from pdf2fb2.timekeeping.workload_units import (
    FLAGGED_SCOPE,
    INSPECTED_UNITS,
    PAGE_UNITS,
    TOKEN_UNITS,
    ConversionWorkload,
    speed_metric_key,
    stage_unit_kind,
)


def measured_units(conversion_workload: ConversionWorkload) -> dict[str, int]:
    """Unit counts that were measured (not estimated) in a run.

    Args:
        conversion_workload: Sizes known at the end of the run.

    Returns:
        Count per unit kind, only for the known kinds.
    """
    known_counts = {PAGE_UNITS: conversion_workload.page_count, TOKEN_UNITS: conversion_workload.token_count,
                    INSPECTED_UNITS: conversion_workload.inspected_token_count}
    return {unit_kind: unit_count for unit_kind, unit_count in known_counts.items() if unit_count is not None}


def measure_text(conversion_workload: ConversionWorkload, proofread_scope: str) -> TextSample:
    """Text density of a finished run.

    Args:
        conversion_workload: Sizes known at the end of the run.
        proofread_scope: ``full`` or ``flagged``; only flagged runs measure the mistake share.

    Returns:
        Measured density (None where not measurable).
    """
    token_count = conversion_workload.token_count
    text_pages = conversion_workload.text_page_count
    tokens_per_page = token_count / text_pages if token_count is not None and text_pages else None
    flagged_share = None
    if proofread_scope == FLAGGED_SCOPE and conversion_workload.inspected_token_count is not None and token_count:
        flagged_share = min(conversion_workload.inspected_token_count / token_count, 1.0)
    return TextSample(tokens_per_page, flagged_share)


class RunTiming:
    """Remaining-time estimate and speed learning of one conversion.

    Usage:
        Created per conversion with the plan and the metrics store; fed with every stage report and with the
        workload sizes as they become known; ``learn_speeds`` stores the run's speeds at the end.
    """

    def __init__(self, conversion_plan: ConversionPlan, metrics_store: SpeedMetricsStore,
                 clock_function: Callable[[], float] = time.monotonic) -> None:
        """Load the stored speeds and start the clock.

        Args:
            conversion_plan: Active stages and proofreading scope.
            metrics_store: Stored speeds of this machine.
            clock_function: Monotonic clock in seconds (injectable for tests).
        """
        self._conversion_plan = conversion_plan
        self._metrics_store = metrics_store
        self._speed_metrics = metrics_store.load_metrics()
        self._stage_clock = StageClock(conversion_plan.stage_names, clock_function)
        self._conversion_workload = ConversionWorkload()

    def record_report(self, stage_name: str, stage_fraction: float) -> None:
        """Take one stage progress report.

        Args:
            stage_name: Reporting stage.
            stage_fraction: Its completed share, 0..1.

        Returns:
            None.
        """
        self._stage_clock.record_report(stage_name, stage_fraction)

    def revise_workload(self, **workload_sizes: int) -> None:
        """Store sizes that became known (``page_count``, ``text_page_count``, ``token_count``,
        ``inspected_token_count``).

        Args:
            **workload_sizes: Known sizes (>= 0).

        Returns:
            None.

        Raises:
            TypeError: For an unknown size name.
        """
        self._conversion_workload = replace(self._conversion_workload, **workload_sizes)

    def remaining_seconds(self) -> float:
        """Pessimistic time left.

        Returns:
            Seconds (>= 0).
        """
        return remaining_run_seconds(self._conversion_plan, self._conversion_workload, self._speed_metrics,
                                     self._stage_clock.stage_states())

    def learn_speeds(self) -> None:
        """Stop the clock and store the speeds measured in the finished stages.

        Returns:
            None.

        Raises:
            OSError: If the metrics file cannot be written.
        """
        self._stage_clock.stop_clock()
        stage_samples = [self._stage_sample(stage_name, stage_record)
                         for stage_name, stage_record in self._stage_clock.finished_records().items()]
        self._speed_metrics = learn_run(
            self._speed_metrics, [stage_sample for stage_sample in stage_samples if stage_sample is not None],
            measure_text(self._conversion_workload, self._conversion_plan.proofread_scope), default_stage_speeds())
        self._metrics_store.save_metrics(self._speed_metrics)

    def _stage_sample(self, stage_name: str, stage_record: StageRecord) -> StageSample | None:
        """Measurement of one finished stage.

        Args:
            stage_name: Finished stage.
            stage_record: Its timing.

        Returns:
            The sample, or None when the stage's unit count was not measured.
        """
        unit_count = measured_units(self._conversion_workload).get(stage_unit_kind(stage_name))
        if unit_count is None:
            return None
        return StageSample(speed_metric_key(stage_name, self._conversion_plan.proofread_scope),
                           stage_record.active_seconds, stage_record.startup_seconds or 0.0, float(unit_count))
