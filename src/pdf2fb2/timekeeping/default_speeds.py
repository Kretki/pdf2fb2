"""Built-in stage speeds used until this machine has measured its own (rough figures for one consumer GPU).

Units: seconds per page, except OCR and verification (per word) and proofreading (per inspected word).
"""

from functools import cache

from pdf2fb2.progress.stage_names import (
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
from pdf2fb2.timekeeping.speed_metrics import RunningStatistic, StageSpeed
from pdf2fb2.timekeeping.workload_units import FLAGGED_SCOPE, FULL_SCOPE, speed_metric_key

# Unmeasured defaults are uncertain: their deviation is half the value, so estimates start on the long side.
DEFAULT_DEVIATION_SHARE = 0.5
# (seconds per unit, startup seconds) per metric key.
DEFAULT_SPEED_TABLE = {
    RENDERING_STAGE: (0.15, 1.0), SEGMENTATION_STAGE: (0.4, 25.0), HANDWRITING_STAGE: (0.3, 15.0),
    CLEANING_STAGE: (0.6, 15.0), RESCAN_SEGMENTATION_STAGE: (0.3, 20.0), PICTURE_STAGE: (0.1, 20.0),
    RESIDUAL_STAGE: (0.1, 20.0), SEGMENT_REVIEW_STAGE: (0.005, 0.0), RECOGNITION_STAGE: (0.08, 40.0),
    REVIEW_STAGE: (0.005, 0.0), STYLING_STAGE: (1.0, 20.0), VERIFICATION_STAGE: (0.02, 20.0),
    DRAWING_STAGE: (0.1, 0.0), ASSEMBLING_STAGE: (0.02, 1.0),
    speed_metric_key(PROOFREADING_STAGE, FULL_SCOPE): (0.1, 60.0),
    speed_metric_key(PROOFREADING_STAGE, FLAGGED_SCOPE): (1.5, 60.0),
}


def uncertain_statistic(default_value: float) -> RunningStatistic:
    """Statistic of a built-in default that has no samples yet.

    Args:
        default_value: Default mean (>= 0).

    Returns:
        Statistic with a deviation of ``DEFAULT_DEVIATION_SHARE`` of the value and no samples.
    """
    return RunningStatistic(default_value, default_value * DEFAULT_DEVIATION_SHARE, 0)


@cache
def default_stage_speeds() -> dict[str, StageSpeed]:
    """Built-in speeds of every stage.

    Returns:
        Speed per metric key (cached: callers must not mutate it).
    """
    return {metric_key: StageSpeed(uncertain_statistic(unit_seconds), uncertain_statistic(startup_seconds))
            for metric_key, (unit_seconds, startup_seconds) in DEFAULT_SPEED_TABLE.items()}
