"""Judge stage: asks the vision-language model (proofread worker venv) whether uncertain units are printed as read."""

from pathlib import Path

from pdf2fb2.progress.progress_contracts import StageProgress
from pdf2fb2.worker_runtime.worker_environment import WorkerEnvironment
from pdf2fb2.worker_runtime.worker_launcher import WorkerFailedError, WorkerLauncher, describe_worker_failure
from worker_protocol.json_files import read_json_document, write_json_document
from worker_protocol.judge_job_schema import JudgeItem, JudgeJob
from worker_protocol.progress_events import JUDGE_PROGRESS_EVENT, MODEL_LOADING_EVENT, read_counter_field

JUDGE_JOB_FILE = "judge_job.json"
JUDGE_RESULT_FILE = "judge_job_result.json"
PROGRESS_STEPS = 1000


class JudgeProgressRelay:
    """Turns judge worker events into ``judging units k/n`` stage progress.

    Usage:
        Passed as the event handler of the judge worker run.
    """

    def __init__(self, stage_progress: StageProgress) -> None:
        """Bind the relay to a stage.

        Args:
            stage_progress: Progress sink of the verification stage.
        """
        self._stage_progress = stage_progress

    def __call__(self, progress_event: dict) -> None:
        """Handle one event; unknown events are ignored.

        Args:
            progress_event: Event decoded by the launcher.

        Returns:
            None.
        """
        if progress_event["event"] == MODEL_LOADING_EVENT:
            self._stage_progress.advance_stage(0, PROGRESS_STEPS, "loading judge model")
            return
        done_count = read_counter_field(progress_event, "done")
        total_count = read_counter_field(progress_event, "total")
        if progress_event["event"] == JUDGE_PROGRESS_EVENT and done_count is not None and total_count:
            self._stage_progress.advance_stage(round(PROGRESS_STEPS * done_count / total_count), PROGRESS_STEPS,
                                               f"judging units {done_count}/{total_count}")


def validated_answers(result_document: dict, judge_items: list[JudgeItem]) -> dict[str, float]:
    """Error probability per item key, checked against the job.

    Args:
        result_document: Decoded result.
        judge_items: Items of the job.

    Returns:
        Key -> probability 0..1.

    Raises:
        WorkerFailedError: If the result does not match the job.
    """
    result_items = result_document.get("items")
    if not isinstance(result_items, list) or len(result_items) != len(judge_items):
        raise WorkerFailedError("Judge worker returned a result that does not match the job.")
    judge_answers = {}
    for judge_item, result_item in zip(judge_items, result_items, strict=True):
        error_value = result_item.get("error_probability") if isinstance(result_item, dict) else None
        if result_item.get("key") != judge_item.item_key or not isinstance(error_value, int | float) \
                or not 0.0 <= error_value <= 1.0:
            raise WorkerFailedError("Judge worker returned a malformed item.")
        judge_answers[judge_item.item_key] = float(error_value)
    return judge_answers


class JudgeStage:
    """Runs the judge worker.

    Usage:
        Built by the pipeline wiring when ``--judge`` is given; used by ``VerificationFeedback``.
    """

    def __init__(self, worker_launcher: WorkerLauncher, worker_environment: WorkerEnvironment,
                 model_directory: Path, device_name: str) -> None:
        """Store the stage configuration.

        Args:
            worker_launcher: Starts the worker process.
            worker_environment: Judge entry of the proofread worker venv.
            model_directory: Vision-language model (``models/qwen3.5-9B``).
            device_name: Device such as ``gpu:0``.
        """
        self._worker_launcher = worker_launcher
        self._worker_environment = worker_environment
        self._model_directory = model_directory
        self._device_name = device_name

    def judge_units(self, judge_items: list[JudgeItem], work_directory: Path,
                    stage_progress: StageProgress) -> dict[str, float]:
        """Error probability of every judged unit.

        Args:
            judge_items: Units with their line crops (paths relative to ``work_directory``).
            work_directory: Output directory.
            stage_progress: Progress sink.

        Returns:
            Item key -> probability that the unit is not printed as read (empty for no items).

        Raises:
            WorkerFailedError: If the worker fails or returns a mismatching result.
        """
        if not judge_items:
            return {}
        judge_job = JudgeJob(work_directory.resolve(), self._model_directory.resolve(), self._device_name,
                             tuple(judge_items), JUDGE_RESULT_FILE)
        job_file = work_directory / JUDGE_JOB_FILE
        write_json_document(job_file, judge_job.to_payload())
        worker_run = self._worker_launcher.launch_worker(self._worker_environment, job_file,
                                                         JudgeProgressRelay(stage_progress))
        if worker_run.exit_code != 0:
            raise WorkerFailedError(describe_worker_failure("Judge worker", worker_run))
        return validated_answers(read_json_document(judge_job.resolve_inside(JUDGE_RESULT_FILE)), judge_items)
