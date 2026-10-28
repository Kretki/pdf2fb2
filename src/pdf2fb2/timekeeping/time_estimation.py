"""Pure time estimates of stages and whole conversions from the stored speeds.

Stage estimates use the upper (mean + deviation) speeds and a safety factor, so they err on the long side.
While a stage runs, its prediction is blended with the extrapolation of its own progress: the more of the stage
is done, the more its own pace counts.
"""

from dataclasses import dataclass

from pdf2fb2.timekeeping.default_speeds import default_stage_speeds
from pdf2fb2.timekeeping.speed_metrics import SpeedMetrics, StageSpeed
from pdf2fb2.timekeeping.workload_units import ConversionWorkload, resolve_workload, speed_metric_key, stage_unit_kind

SAFETY_FACTOR = 1.1


@dataclass(frozen=True)
class ConversionPlan:
    """Stages a conversion runs and how it proofreads.

    Attributes:
        stage_names: Active progress stages in execution order.
        proofread_scope: ``full`` or ``flagged``.
    """

    stage_names: tuple[str, ...]
    proofread_scope: str


@dataclass(frozen=True)
class StageState:
    """How far a running (or finished) stage is.

    Attributes:
        active_seconds: Time spent in the stage so far.
        stage_fraction: Completed share reported by the stage, 0..1.
        is_finished: Whether the stage is over.
    """

    active_seconds: float = 0.0
    stage_fraction: float = 0.0
    is_finished: bool = False


def stage_speed_for(speed_metrics: SpeedMetrics, metric_key: str) -> StageSpeed | None:
    """Stored speed of a stage, or its built-in default.

    Args:
        speed_metrics: Stored metrics.
        metric_key: Key of the stage.

    Returns:
        The speed, or None for a stage without any figure.
    """
    return speed_metrics.stage_speeds.get(metric_key) or default_stage_speeds().get(metric_key)


def predicted_stage_seconds(stage_name: str, conversion_plan: ConversionPlan, conversion_workload: ConversionWorkload,
                            speed_metrics: SpeedMetrics) -> float:
    """Pessimistic duration of one whole stage.

    Args:
        stage_name: Progress stage name.
        conversion_plan: Active stages and proofreading scope.
        conversion_workload: Known sizes (unknown ones are estimated pessimistically).
        speed_metrics: Stored metrics.

    Returns:
        Seconds (>= 0; 0 for a stage without figures).
    """
    stage_speed = stage_speed_for(speed_metrics, speed_metric_key(stage_name, conversion_plan.proofread_scope))
    if stage_speed is None:
        return 0.0
    resolved_workload = resolve_workload(conversion_workload, speed_metrics.text_profile(),
                                         conversion_plan.proofread_scope)
    unit_count = resolved_workload.unit_counts[stage_unit_kind(stage_name)]
    return SAFETY_FACTOR * (stage_speed.startup_seconds.upper_value()
                            + stage_speed.unit_seconds.upper_value() * unit_count)


def remaining_stage_seconds(predicted_seconds: float, stage_state: StageState) -> float:
    """Remaining time of one stage, blending its prediction with its own pace.

    Args:
        predicted_seconds: Pessimistic duration of the whole stage.
        stage_state: Time spent and completed share.

    Returns:
        Seconds left (>= 0; 0 once finished).
    """
    if stage_state.is_finished:
        return 0.0
    stage_fraction = min(max(stage_state.stage_fraction, 0.0), 1.0)
    if stage_fraction <= 0.0:
        return max(predicted_seconds - stage_state.active_seconds, 0.0)
    projected_seconds = stage_state.active_seconds / stage_fraction
    blended_seconds = (1.0 - stage_fraction) * predicted_seconds + stage_fraction * projected_seconds
    return max(blended_seconds - stage_state.active_seconds, 0.0)


def remaining_run_seconds(conversion_plan: ConversionPlan, conversion_workload: ConversionWorkload,
                          speed_metrics: SpeedMetrics, stage_states: dict[str, StageState]) -> float:
    """Remaining time of a whole conversion.

    Usage:
        Called on every progress update; with empty ``stage_states`` it is the start-of-run estimate.

    Args:
        conversion_plan: Active stages and proofreading scope.
        conversion_workload: Known sizes.
        speed_metrics: Stored metrics.
        stage_states: State of every stage started so far (missing: not started).

    Returns:
        Seconds left (>= 0).
    """
    return sum(remaining_stage_seconds(
        predicted_stage_seconds(stage_name, conversion_plan, conversion_workload, speed_metrics),
        stage_states.get(stage_name, StageState())) for stage_name in conversion_plan.stage_names)
