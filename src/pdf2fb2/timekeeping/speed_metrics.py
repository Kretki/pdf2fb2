"""Hardware speed metrics: running statistics of every stage's speed and of the text density, and their update.

Every statistic keeps an exponential moving mean and mean absolute deviation; estimates use the upper value
(mean + deviation), so the shown time errs on the long side. Samples far below the mean (a stage served from the
OCR cache, an empty book) are not learned from.
"""

from dataclasses import dataclass, field, replace

from pdf2fb2.timekeeping.workload_units import TextProfile

LEARNING_RATE = 0.3
FAST_OUTLIER_RATIO = 0.25
MIN_SAMPLES_FOR_OUTLIERS = 3
MIN_UNITS_PER_SAMPLE = 1.0


@dataclass(frozen=True)
class RunningStatistic:
    """Moving mean and mean absolute deviation of a positive quantity.

    Attributes:
        mean_value: Moving mean (>= 0).
        mean_deviation: Moving mean absolute deviation (>= 0).
        sample_count: Number of samples learned so far.
    """

    mean_value: float
    mean_deviation: float = 0.0
    sample_count: int = 0

    def upper_value(self) -> float:
        """Pessimistic value of the statistic.

        Returns:
            ``mean_value + mean_deviation``.
        """
        return self.mean_value + self.mean_deviation


def learn_sample(running_statistic: RunningStatistic, sample_value: float) -> RunningStatistic:
    """Fold one sample into a statistic.

    Usage:
        Called once per finished run for every measured quantity. The first samples are averaged, later ones
        move the mean by ``LEARNING_RATE``; a sample below a quarter of an established mean is ignored.

    Args:
        running_statistic: Current statistic (sample_count 0: only the built-in default).
        sample_value: New measurement (>= 0).

    Returns:
        The updated statistic (unchanged for an ignored sample).
    """
    if running_statistic.sample_count >= MIN_SAMPLES_FOR_OUTLIERS \
            and sample_value < FAST_OUTLIER_RATIO * running_statistic.mean_value:
        return running_statistic
    if running_statistic.sample_count == 0:
        return RunningStatistic(sample_value, abs(sample_value - running_statistic.mean_value) / 2, 1)
    learning_weight = max(1.0 / (running_statistic.sample_count + 1), LEARNING_RATE)
    sample_deviation = abs(sample_value - running_statistic.mean_value)
    return RunningStatistic(
        mean_value=running_statistic.mean_value + learning_weight * (sample_value - running_statistic.mean_value),
        mean_deviation=running_statistic.mean_deviation
        + learning_weight * (sample_deviation - running_statistic.mean_deviation),
        sample_count=running_statistic.sample_count + 1)


@dataclass(frozen=True)
class StageSpeed:
    """Speed of one stage on this machine.

    Attributes:
        unit_seconds: Seconds per unit (page or token) once the stage is running.
        startup_seconds: Seconds before the first unit is done (model loading, worker start).
    """

    unit_seconds: RunningStatistic
    startup_seconds: RunningStatistic


@dataclass(frozen=True)
class StageSample:
    """One measured stage of a finished run.

    Attributes:
        metric_key: Key of the stage in the metrics (see ``speed_metric_key``).
        active_seconds: Total time the stage was running.
        startup_seconds: Time until its first unit was done.
        unit_count: Units it processed.
    """

    metric_key: str
    active_seconds: float
    startup_seconds: float
    unit_count: float


@dataclass(frozen=True)
class TextSample:
    """Text density measured in a finished run; None where not measured.

    Attributes:
        tokens_per_text_page: Words per page with text.
        flagged_token_share: Share of words the verifier flagged.
    """

    tokens_per_text_page: float | None = None
    flagged_token_share: float | None = None


@dataclass(frozen=True)
class SpeedMetrics:
    """Everything stored about this machine's speed.

    Attributes:
        stage_speeds: Speed per metric key.
        tokens_per_text_page: Words on a text page.
        flagged_token_share: Share of words flagged as possible mistakes.
        run_count: Number of runs learned from.
    """

    stage_speeds: dict[str, StageSpeed] = field(default_factory=dict)
    tokens_per_text_page: RunningStatistic = RunningStatistic(350.0)
    flagged_token_share: RunningStatistic = RunningStatistic(0.1)
    run_count: int = 0

    def text_profile(self) -> TextProfile:
        """Pessimistic text density.

        Returns:
            Upper words per text page and upper flagged share (capped at 1).
        """
        return TextProfile(max(self.tokens_per_text_page.upper_value(), 1.0),
                           min(self.flagged_token_share.upper_value(), 1.0))


def learn_stage(stage_speed: StageSpeed, stage_sample: StageSample) -> StageSpeed:
    """Fold one stage measurement into its speed.

    Args:
        stage_speed: Current speed of the stage.
        stage_sample: Measurement of a finished run.

    Returns:
        Updated speed; the per-unit speed is learned only when the stage processed at least one unit.
    """
    learned_startup = learn_sample(stage_speed.startup_seconds, stage_sample.startup_seconds)
    if stage_sample.unit_count < MIN_UNITS_PER_SAMPLE:
        return replace(stage_speed, startup_seconds=learned_startup)
    working_seconds = max(stage_sample.active_seconds - stage_sample.startup_seconds, 0.0)
    return StageSpeed(learn_sample(stage_speed.unit_seconds, working_seconds / stage_sample.unit_count),
                      learned_startup)


def learn_text(speed_metrics: SpeedMetrics, text_sample: TextSample) -> SpeedMetrics:
    """Fold the text density of a run into the metrics.

    Args:
        speed_metrics: Current metrics.
        text_sample: Measured density (None fields are skipped).

    Returns:
        Updated metrics.
    """
    tokens_statistic = speed_metrics.tokens_per_text_page if text_sample.tokens_per_text_page is None \
        else learn_sample(speed_metrics.tokens_per_text_page, text_sample.tokens_per_text_page)
    flagged_statistic = speed_metrics.flagged_token_share if text_sample.flagged_token_share is None \
        else learn_sample(speed_metrics.flagged_token_share, text_sample.flagged_token_share)
    return replace(speed_metrics, tokens_per_text_page=tokens_statistic, flagged_token_share=flagged_statistic)


def learn_run(speed_metrics: SpeedMetrics, stage_samples: list[StageSample], text_sample: TextSample,
              default_speeds: dict[str, StageSpeed]) -> SpeedMetrics:
    """Fold every measurement of a finished run into the metrics.

    Usage:
        Called at the end of every conversion; the result is saved for the next run.

    Args:
        speed_metrics: Current metrics.
        stage_samples: Measured stages.
        text_sample: Measured text density.
        default_speeds: Built-in speeds of stages never measured before.

    Returns:
        Updated metrics with ``run_count`` increased.
    """
    stage_speeds = dict(speed_metrics.stage_speeds)
    for stage_sample in stage_samples:
        current_speed = stage_speeds.get(stage_sample.metric_key) or default_speeds.get(stage_sample.metric_key)
        if current_speed is not None:
            stage_speeds[stage_sample.metric_key] = learn_stage(current_speed, stage_sample)
    return replace(learn_text(speed_metrics, text_sample), stage_speeds=stage_speeds,
                   run_count=speed_metrics.run_count + 1)
