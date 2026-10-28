"""Verification stage: delegates unit labelling of OCR texts to the verifier worker venv."""

import dataclasses
from dataclasses import dataclass
from pathlib import Path

from pdf2fb2.progress.progress_contracts import StageProgress
from pdf2fb2.worker_runtime.worker_environment import WorkerEnvironment
from pdf2fb2.worker_runtime.worker_launcher import WorkerFailedError, WorkerLauncher, describe_worker_failure
from worker_protocol.json_files import read_json_document, write_json_document
from worker_protocol.progress_events import MODEL_LOADING_EVENT, VERIFY_PROGRESS_EVENT, read_counter_field
from worker_protocol.verify_job_schema import LABEL_NAMES, MAX_UNIT_LENGTH, VerifyItem, VerifyJob

PROGRESS_STEPS = 1000


@dataclass(frozen=True)
class VerifyOutcome:
    """Labels of one verification job.

    Attributes:
        unit_probabilities: Item key -> ``[correct, misread, inserted]`` per unit.
        class_thresholds: ``{"misread": float, "inserted": float}`` tuned for the model.
    """

    unit_probabilities: dict[str, list[list[float]]]
    class_thresholds: dict[str, float]


class VerifyProgressRelay:
    """Turns verifier worker events into ``verifying text k/n`` stage progress.

    Usage:
        Passed as the event handler of the verifier worker run.
    """

    def __init__(self, stage_progress: StageProgress, phase_label: str) -> None:
        """Bind the relay to a stage.

        Args:
            stage_progress: Progress sink of the verification stage.
            phase_label: Text shown before the counter.
        """
        self._stage_progress = stage_progress
        self._phase_label = phase_label

    def __call__(self, progress_event: dict) -> None:
        """Handle one event; unknown events are ignored.

        Args:
            progress_event: Event decoded by the launcher.

        Returns:
            None.
        """
        if progress_event["event"] == MODEL_LOADING_EVENT:
            self._stage_progress.advance_stage(0, PROGRESS_STEPS, "loading token verifier")
            return
        done_count = read_counter_field(progress_event, "done")
        total_count = read_counter_field(progress_event, "total")
        if progress_event["event"] == VERIFY_PROGRESS_EVENT and done_count is not None and total_count:
            self._stage_progress.advance_stage(round(PROGRESS_STEPS * done_count / total_count), PROGRESS_STEPS,
                                               f"{self._phase_label} {done_count}/{total_count}")


def validate_outcome(worker_result: object, verify_items: tuple[VerifyItem, ...]) -> VerifyOutcome:
    """Check the worker result against the job.

    Args:
        worker_result: Decoded result document.
        verify_items: Items of the job.

    Returns:
        The outcome.

    Raises:
        WorkerFailedError: If items, unit counts, probabilities or thresholds do not match.
    """
    result_items = worker_result.get("items") if isinstance(worker_result, dict) else None
    thresholds = worker_result.get("thresholds") if isinstance(worker_result, dict) else None
    if not isinstance(result_items, list) or len(result_items) != len(verify_items) or not isinstance(thresholds, dict):
        raise WorkerFailedError("Verifier worker returned items that do not match the job.")
    for result_item, verify_item in zip(result_items, verify_items, strict=True):
        probability_rows = result_item.get("probabilities") if isinstance(result_item, dict) else None
        if result_item.get("key") != verify_item.item_key or not isinstance(probability_rows, list) \
                or len(probability_rows) != len(verify_item.item_units) or not all(
                    isinstance(probability_row, list) and len(probability_row) == len(LABEL_NAMES)
                    and all(isinstance(class_value, (int, float)) for class_value in probability_row)
                    for probability_row in probability_rows):
            raise WorkerFailedError(f"Verifier worker returned a malformed item {verify_item.item_key!r}.")
    if not all(isinstance(thresholds.get(class_name), (int, float)) for class_name in LABEL_NAMES[1:]):
        raise WorkerFailedError("Verifier worker returned malformed thresholds.")
    return VerifyOutcome({result_item["key"]: result_item["probabilities"] for result_item in result_items},
                         {class_name: float(thresholds[class_name]) for class_name in LABEL_NAMES[1:]})


class VerifyStage:
    """Labels OCR text units by running the verifier worker in its own venv.

    Usage:
        Build once with injected launcher/environment and call ``label_units`` (see ``UnitLabeler``).
    """

    def __init__(self, worker_launcher: WorkerLauncher, worker_environment: WorkerEnvironment,
                 model_directory: Path, device_name: str) -> None:
        """Store the stage configuration.

        Args:
            worker_launcher: Starts the worker process.
            worker_environment: Verifier worker venv description.
            model_directory: Fine-tuned verifier (``models/token_verifier``).
            device_name: Device such as ``gpu:0`` or ``cpu``.
        """
        self._worker_launcher = worker_launcher
        self._worker_environment = worker_environment
        self._model_directory = model_directory
        self._device_name = device_name

    def label_units(self, unit_lists: dict[str, list[str]], work_directory: Path, job_name: str,
                    stage_progress: StageProgress) -> VerifyOutcome:
        """Probabilities of every unit of every text.

        Args:
            unit_lists: Item key -> units (units longer than the protocol limit are cut).
            work_directory: Output directory for the job and result files.
            job_name: File stem of the job, e.g. ``verify_job`` (result ``<stem>_result.json``).
            stage_progress: Receives progress.

        Returns:
            Validated outcome.

        Raises:
            WorkerFailedError: If the worker fails or returns a mismatching result.
        """
        plain_items = [VerifyItem(item_key, tuple(item_units)) for item_key, item_units in unit_lists.items()]
        return self.label_items(plain_items, work_directory, job_name, stage_progress)

    def label_items(self, verify_items: list[VerifyItem], work_directory: Path, job_name: str,
                    stage_progress: StageProgress) -> VerifyOutcome:
        """Probabilities of every unit of items that may carry context, page evidence and crops.

        Args:
            verify_items: Items (empty ones are skipped; units and context longer than the limit are cut).
            work_directory: Output directory for the job and result files (crop paths are relative to it).
            job_name: File stem of the job (result ``<stem>_result.json``).
            stage_progress: Receives progress.

        Returns:
            Validated outcome.

        Raises:
            WorkerFailedError: If the worker fails or returns a mismatching result.
        """
        verify_items = tuple(dataclasses.replace(
            verify_item, item_units=tuple(unit_text[:MAX_UNIT_LENGTH] for unit_text in verify_item.item_units),
            context_before=tuple(unit_text[:MAX_UNIT_LENGTH] for unit_text in verify_item.context_before),
            context_after=tuple(unit_text[:MAX_UNIT_LENGTH] for unit_text in verify_item.context_after))
            for verify_item in verify_items if verify_item.item_units)
        if not verify_items:
            return VerifyOutcome({}, {"misread": 1.0, "inserted": 1.0})
        verify_job = VerifyJob(work_directory.resolve(), self._model_directory.resolve(), self._device_name,
                               verify_items, f"{job_name}_result.json")
        job_file = work_directory / f"{job_name}.json"
        write_json_document(job_file, verify_job.to_payload())
        worker_run = self._worker_launcher.launch_worker(self._worker_environment, job_file,
                                                         VerifyProgressRelay(stage_progress, "verifying text"))
        if worker_run.exit_code != 0:
            raise WorkerFailedError(describe_worker_failure("Verifier worker", worker_run))
        return validate_outcome(read_json_document(verify_job.resolve_inside(verify_job.result_file)), verify_items)
