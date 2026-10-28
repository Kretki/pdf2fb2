"""JSON file that keeps this machine's speed metrics between runs (``metrics/speed_metrics.json``).

The file is read on every start and rewritten at the end of every conversion. A missing, unreadable or malformed
file falls back to the built-in defaults; every number read back is bounded.
"""

import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from pdf2fb2.timekeeping.speed_metrics import RunningStatistic, SpeedMetrics, StageSpeed

MAX_METRIC_VALUE = 1e7
MAX_SAMPLE_COUNT = 10**9
MAX_KEY_LENGTH = 64
MAX_FILE_BYTES = 1_000_000
FORMAT_VERSION = 1


class SpeedMetricsStore(Protocol):
    """Loads and saves the speed metrics.

    Usage:
        Injected into the progress tracking and the start-of-run estimate.
    """

    def load_metrics(self) -> SpeedMetrics:
        """Read the stored metrics.

        Returns:
            Stored metrics, or defaults when nothing usable is stored.
        """
        ...

    def save_metrics(self, speed_metrics: SpeedMetrics) -> None:
        """Store the metrics.

        Args:
            speed_metrics: Metrics to keep for the next run.

        Returns:
            None.
        """
        ...


def bounded_number(raw_value: object, upper_bound: float) -> float:
    """Validate a number read from the file.

    Args:
        raw_value: Decoded JSON value.
        upper_bound: Largest accepted value.

    Returns:
        The value as float.

    Raises:
        ValueError: If it is not a finite number in 0..upper_bound.
    """
    if isinstance(raw_value, bool) or not isinstance(raw_value, int | float) or not math.isfinite(raw_value) \
            or not 0 <= raw_value <= upper_bound:
        raise ValueError(f"bad metric value {raw_value!r}")
    return float(raw_value)


def decode_statistic(raw_statistic: object) -> RunningStatistic:
    """Decode one running statistic.

    Args:
        raw_statistic: ``{"mean_value", "mean_deviation", "sample_count"}`` mapping.

    Returns:
        The statistic.

    Raises:
        ValueError: If a field is missing or out of range.
    """
    if not isinstance(raw_statistic, dict):
        raise ValueError("statistic is not an object")
    return RunningStatistic(bounded_number(raw_statistic.get("mean_value"), MAX_METRIC_VALUE),
                            bounded_number(raw_statistic.get("mean_deviation"), MAX_METRIC_VALUE),
                            int(bounded_number(raw_statistic.get("sample_count"), MAX_SAMPLE_COUNT)))


def decode_stage_speeds(raw_speeds: object) -> dict[str, StageSpeed]:
    """Decode the per-stage speeds.

    Args:
        raw_speeds: Mapping of metric key to ``{"unit_seconds", "startup_seconds"}``.

    Returns:
        Speed per metric key.

    Raises:
        ValueError: If the mapping or one of its entries is malformed.
    """
    if not isinstance(raw_speeds, dict):
        raise ValueError("stage_speeds is not an object")
    stage_speeds = {}
    for metric_key, raw_speed in raw_speeds.items():
        if not isinstance(metric_key, str) or len(metric_key) > MAX_KEY_LENGTH or not isinstance(raw_speed, dict):
            raise ValueError(f"bad stage entry {metric_key!r}")
        stage_speeds[metric_key] = StageSpeed(decode_statistic(raw_speed.get("unit_seconds")),
                                              decode_statistic(raw_speed.get("startup_seconds")))
    return stage_speeds


def decode_metrics(raw_document: object) -> SpeedMetrics:
    """Decode the whole metrics document.

    Args:
        raw_document: Parsed JSON.

    Returns:
        The metrics.

    Raises:
        ValueError: If the document is malformed.
    """
    if not isinstance(raw_document, dict) or raw_document.get("format_version") != FORMAT_VERSION:
        raise ValueError("unknown metrics format")
    return SpeedMetrics(stage_speeds=decode_stage_speeds(raw_document.get("stage_speeds")),
                        tokens_per_text_page=decode_statistic(raw_document.get("tokens_per_text_page")),
                        flagged_token_share=decode_statistic(raw_document.get("flagged_token_share")),
                        run_count=int(bounded_number(raw_document.get("run_count"), MAX_SAMPLE_COUNT)))


class JsonSpeedMetricsStore:
    """``SpeedMetricsStore`` backed by one JSON file.

    Usage:
        Built by the command-line wiring with ``project_paths.SPEED_METRICS_FILE``.
    """

    def __init__(self, metrics_path: Path) -> None:
        """Bind the store to its file.

        Args:
            metrics_path: JSON file (its directory is created on the first save).
        """
        self._metrics_path = metrics_path

    def load_metrics(self) -> SpeedMetrics:
        """Read the stored metrics.

        Returns:
            Stored metrics, or ``SpeedMetrics()`` defaults for a missing, oversized or malformed file.
        """
        try:
            if self._metrics_path.stat().st_size > MAX_FILE_BYTES:
                return SpeedMetrics()
            return decode_metrics(json.loads(self._metrics_path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            return SpeedMetrics()

    def save_metrics(self, speed_metrics: SpeedMetrics) -> None:
        """Write the metrics atomically (temporary file, then rename).

        Args:
            speed_metrics: Metrics to keep.

        Returns:
            None.

        Raises:
            OSError: If the file cannot be written.
        """
        self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self._metrics_path.with_suffix(".json.tmp")
        temporary_path.write_text(json.dumps({"format_version": FORMAT_VERSION, **asdict(speed_metrics)},
                                             indent=1, sort_keys=True), encoding="utf-8")
        os.replace(temporary_path, self._metrics_path)


class TransientSpeedMetricsStore:
    """``SpeedMetricsStore`` that keeps nothing: built-in defaults in, measurements dropped.

    Usage:
        Default of pipelines built without a metrics file (tests, embedding).
    """

    def load_metrics(self) -> SpeedMetrics:
        """Return the built-in defaults.

        Returns:
            ``SpeedMetrics()``.
        """
        return SpeedMetrics()

    def save_metrics(self, speed_metrics: SpeedMetrics) -> None:
        """Drop the metrics.

        Args:
            speed_metrics: Ignored.

        Returns:
            None.
        """
