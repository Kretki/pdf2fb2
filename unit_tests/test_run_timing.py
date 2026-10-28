"""Unit tests of the speed metrics, time estimates, stage clock, parallel mapping and page counting."""

from pathlib import Path

from pdf2fb2.document_sources.page_counting import count_book_pages, count_djvu_pages
from pdf2fb2.parallel_work.parallel_mapping import map_in_threads
from pdf2fb2.progress.pipeline_progress import PipelineProgress
from pdf2fb2.progress.stage_names import ASSEMBLING_STAGE, PROOFREADING_STAGE, RECOGNITION_STAGE, RENDERING_STAGE
from pdf2fb2.timekeeping.run_estimate import estimate_book_seconds
from pdf2fb2.timekeeping.run_timing import RunTiming
from pdf2fb2.timekeeping.speed_metrics import RunningStatistic, SpeedMetrics, learn_sample
from pdf2fb2.timekeeping.speed_metrics_store import JsonSpeedMetricsStore
from pdf2fb2.timekeeping.stage_clock import StageClock
from pdf2fb2.timekeeping.time_estimation import ConversionPlan, StageState, remaining_stage_seconds
from pdf2fb2.timekeeping.workload_units import FLAGGED_SCOPE, FULL_SCOPE


class ManualClock:
    """Clock the test moves by hand."""

    def __init__(self) -> None:
        """Start at zero."""
        self.current_seconds = 0.0

    def __call__(self) -> float:
        """Current time.

        Returns:
            Seconds.
        """
        return self.current_seconds


class RecordingDisplay:
    """Display that keeps every remaining-time value it is shown."""

    def __init__(self) -> None:
        """Start empty."""
        self.remaining_values: list[float | None] = []

    def show_progress(self, overall_fraction: float, status_text: str, remaining_seconds: float | None = None) -> None:
        """Keep the remaining time.

        Args:
            overall_fraction: Ignored.
            status_text: Ignored.
            remaining_seconds: Kept.
        """
        self.remaining_values.append(remaining_seconds)

    def show_message(self, message_text: str) -> None:
        """Ignore messages.

        Args:
            message_text: Ignored.
        """

    def close_display(self) -> None:
        """Nothing to close."""


def test_learn_sample_averages_then_ignores_fast_outliers() -> None:
    """The first samples are averaged; a sample far below an established mean (cache hit) is ignored."""
    learned_statistic = RunningStatistic(1.0)
    for sample_value in (2.0, 2.0, 2.0):
        learned_statistic = learn_sample(learned_statistic, sample_value)
    assert abs(learned_statistic.mean_value - 2.0) < 1e-9
    assert learn_sample(learned_statistic, 0.01) == learned_statistic
    assert learned_statistic.upper_value() >= learned_statistic.mean_value


def test_remaining_stage_blends_prediction_with_pace() -> None:
    """Without progress the prediction counts; with progress the stage's own pace counts more."""
    assert remaining_stage_seconds(100.0, StageState(30.0, 0.0)) == 70.0
    assert remaining_stage_seconds(100.0, StageState(30.0, 0.0, True)) == 0.0
    halfway_seconds = remaining_stage_seconds(100.0, StageState(100.0, 0.5))
    assert 0.0 < halfway_seconds < 100.0


def test_flagged_proofreading_estimate_grows_with_pages() -> None:
    """Estimates grow with the page count and differ between proofreading scopes."""
    stage_names = (RENDERING_STAGE, RECOGNITION_STAGE, PROOFREADING_STAGE, ASSEMBLING_STAGE)
    flagged_plan = ConversionPlan(stage_names, FLAGGED_SCOPE)
    full_plan = ConversionPlan(stage_names, FULL_SCOPE)
    assert estimate_book_seconds(flagged_plan, 100, SpeedMetrics()) > estimate_book_seconds(flagged_plan, 10,
                                                                                            SpeedMetrics())
    assert estimate_book_seconds(full_plan, 100, SpeedMetrics()) != estimate_book_seconds(flagged_plan, 100,
                                                                                         SpeedMetrics())


def test_stage_clock_sums_reentered_stage() -> None:
    """Assembling runs before and after proofreading: its time is summed and it finishes last."""
    test_clock = ManualClock()
    stage_clock = StageClock((PROOFREADING_STAGE, ASSEMBLING_STAGE), test_clock)
    stage_clock.record_report(ASSEMBLING_STAGE, 0.0)
    test_clock.current_seconds = 2.0
    stage_clock.record_report(PROOFREADING_STAGE, 0.5)
    assert not stage_clock.stage_states()[PROOFREADING_STAGE].is_finished
    test_clock.current_seconds = 12.0
    stage_clock.record_report(ASSEMBLING_STAGE, 1.0)
    test_clock.current_seconds = 13.0
    stage_clock.stop_clock()
    finished_records = stage_clock.finished_records()
    assert finished_records[ASSEMBLING_STAGE].active_seconds == 3.0
    assert finished_records[PROOFREADING_STAGE].active_seconds == 10.0


def test_run_timing_learns_and_stores_speeds(tmp_path: Path) -> None:
    """A finished run writes the metrics file; the next run loads the learned speeds."""
    test_clock = ManualClock()
    metrics_store = JsonSpeedMetricsStore(tmp_path / "speed_metrics.json")
    conversion_plan = ConversionPlan((RENDERING_STAGE, ASSEMBLING_STAGE), FULL_SCOPE)
    recording_display = RecordingDisplay()
    pipeline_progress = PipelineProgress(recording_display, list(conversion_plan.stage_names),
                                         RunTiming(conversion_plan, metrics_store, test_clock), test_clock)
    pipeline_progress.revise_workload(page_count=10, text_page_count=10, token_count=3000)
    pipeline_progress.stage_tracker(RENDERING_STAGE).advance_stage(0, 10, "rendering")
    test_clock.current_seconds = 0.5
    pipeline_progress.stage_tracker(RENDERING_STAGE).advance_stage(1, 10, "rendering")
    test_clock.current_seconds = 5.0
    pipeline_progress.stage_tracker(RENDERING_STAGE).advance_stage(10, 10, "rendered")
    pipeline_progress.stage_tracker(ASSEMBLING_STAGE).advance_stage(1, 1, "assembled")
    pipeline_progress.close_progress(True)
    learned_metrics = metrics_store.load_metrics()
    assert learned_metrics.run_count == 1
    assert abs(learned_metrics.stage_speeds[RENDERING_STAGE].unit_seconds.mean_value - 0.45) < 1e-9
    assert abs(learned_metrics.tokens_per_text_page.mean_value - 300.0) < 1e-9
    assert all(remaining_value is not None for remaining_value in recording_display.remaining_values)


def test_malformed_metrics_file_falls_back_to_defaults(tmp_path: Path) -> None:
    """A broken metrics file never stops a run."""
    metrics_path = tmp_path / "speed_metrics.json"
    metrics_path.write_text('{"format_version": 1, "run_count": -5}', encoding="utf-8")
    assert JsonSpeedMetricsStore(metrics_path).load_metrics() == SpeedMetrics()


def test_map_in_threads_keeps_order() -> None:
    """Results come back in input order and progress counts every item."""
    progress_counts: list[int] = []
    squared_values = map_in_threads(lambda item_value: item_value * item_value, list(range(20)),
                                    lambda finished_count, item_count: progress_counts.append(finished_count))
    assert squared_values == [item_value * item_value for item_value in range(20)]
    assert sorted(progress_counts) == list(range(1, 21))


def test_djvu_page_count_from_header(tmp_path: Path) -> None:
    """Single-page and multi-page DjVu headers are counted; other files give None."""
    single_page = tmp_path / "single.djvu"
    single_page.write_bytes(b"AT&TFORM\x00\x00\x00\x10DJVUINFO" + bytes(16))
    multi_page = tmp_path / "multi.djvu"
    multi_page.write_bytes(b"AT&TFORM\x00\x00\x01\x00DJVMDIRM\x00\x00\x00\x20\x81\x00\x2a" + bytes(16))
    broken_file = tmp_path / "broken.djvu"
    broken_file.write_bytes(b"not a djvu file at all, really not")
    assert count_djvu_pages(single_page) == 1
    assert count_djvu_pages(multi_page) == 42
    assert count_book_pages(broken_file) is None
