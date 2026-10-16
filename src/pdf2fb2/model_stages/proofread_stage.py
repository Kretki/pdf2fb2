"""Proofreading stage: delegates language-model scoring of the book text to the proofread worker venv."""

from pathlib import Path

from pdf2fb2.progress.progress_contracts import StageProgress
from pdf2fb2.worker_runtime.worker_environment import WorkerEnvironment
from pdf2fb2.worker_runtime.worker_launcher import WorkerFailedError, WorkerLauncher, describe_worker_failure
from worker_protocol.json_files import read_json_document, write_json_document
from worker_protocol.progress_events import MODEL_LOADING_EVENT, PROOFREAD_PROGRESS_EVENT, read_counter_field
from worker_protocol.proofread_job_schema import MAX_WORD_LENGTH, ProofreadJob, ProofreadSettings, TextBlock

PROOFREAD_JOB_FILE = "proofread_job.json"
PROOFREAD_RESULT_FILE = "proofread_result.json"
CHECK_FIELDS = {"word_checks": ("keep", "drop"), "block_checks": ("keep", "drop_separate", "drop_merged"),
                "break_checks": ("keep", "merged")}
DEFAULT_SETTINGS = ProofreadSettings(candidate_logprob=-9.0, context_characters=1200, continuation_words=12,
                                     max_word_checks=50_000)
# Share of the stage given to scoring the whole text; the deletion tests take the rest.
SCORING_SHARE = 0.3
PROGRESS_STEPS = 1000


class ProofreadProgressRelay:
    """Turns proofread worker events into ``proofreading: scoring/checking k/n`` stage progress.

    Usage:
        Passed as the event handler of the proofread worker run.
    """

    def __init__(self, stage_progress: StageProgress) -> None:
        """Bind the relay to a stage.

        Args:
            stage_progress: Progress sink of the proofreading stage.
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
            self._stage_progress.advance_stage(0, PROGRESS_STEPS, "loading language model")
            return
        done_count = read_counter_field(progress_event, "done")
        total_count = read_counter_field(progress_event, "total")
        if progress_event["event"] != PROOFREAD_PROGRESS_EVENT or done_count is None or not total_count:
            return
        scoring_phase = progress_event.get("phase") == "scoring"
        phase_fraction = done_count / total_count
        stage_fraction = SCORING_SHARE * phase_fraction if scoring_phase \
            else SCORING_SHARE + (1 - SCORING_SHARE) * phase_fraction
        phase_label = "scoring text" if scoring_phase else "testing suspicious spots"
        self._stage_progress.advance_stage(round(PROGRESS_STEPS * stage_fraction), PROGRESS_STEPS,
                                           f"proofreading: {phase_label} {done_count}/{total_count}")


def validate_result(worker_result: object, text_blocks: tuple[TextBlock, ...]) -> dict:
    """Check that the worker result matches the job.

    Args:
        worker_result: Decoded result document.
        text_blocks: Blocks of the job.

    Returns:
        The result.

    Raises:
        WorkerFailedError: If blocks, words or test entries do not match.
    """
    result_blocks = worker_result.get("blocks") if isinstance(worker_result, dict) else None
    if not isinstance(result_blocks, list) or len(result_blocks) != len(text_blocks) or any(
            not isinstance(result_block, dict) or len(result_block.get("words") or []) != len(text_block.block_words)
            for result_block, text_block in zip(result_blocks, text_blocks, strict=True)):
        raise WorkerFailedError("Proofread worker returned blocks that do not match the job.")
    for check_group, score_fields in CHECK_FIELDS.items():
        group_entries = worker_result.get(check_group)
        if not isinstance(group_entries, list) or not all(
                isinstance(check_entry, dict) and isinstance(check_entry.get("block"), int)
                and 0 <= check_entry["block"] < len(text_blocks)
                and all(isinstance(check_entry.get(score_field), (int, float)) for score_field in score_fields)
                for check_entry in group_entries):
            raise WorkerFailedError(f"Proofread worker returned malformed {check_group}.")
    return worker_result


class ProofreadStage:
    """Scores text blocks with the local language model by running the proofread worker in its own venv.

    Usage:
        Build once with injected launcher/environment and call ``score_blocks``.
    """

    def __init__(self, worker_launcher: WorkerLauncher, worker_environment: WorkerEnvironment,
                 model_directory: Path, device_name: str) -> None:
        """Store the stage configuration.

        Args:
            worker_launcher: Starts the worker process.
            worker_environment: Proofread worker venv description.
            model_directory: Directory of the Qwen3.5 weights.
            device_name: Device such as ``gpu:0`` or ``cpu``.
        """
        self._worker_launcher = worker_launcher
        self._worker_environment = worker_environment
        self._model_directory = model_directory
        self._device_name = device_name

    def score_blocks(self, block_words: list[tuple[str, tuple[str, ...]]], block_checks: list[int],
                     break_checks: list[int], work_directory: Path, stage_progress: StageProgress) -> dict:
        """Score every token and run the deletion tests.

        Args:
            block_words: ``(kind, plain words)`` per block in reading order.
            block_checks: Blocks to test for deletion.
            break_checks: Blocks whose break from the previous block is tested.
            work_directory: Output directory for the job and result files.
            stage_progress: Receives proofreading progress.

        Returns:
            Validated worker result (see ``worker_protocol.proofread_job_schema``).

        Raises:
            WorkerFailedError: If the worker fails or returns a mismatching result.
        """
        text_blocks = tuple(TextBlock(block_kind, tuple(word_text[:MAX_WORD_LENGTH] for word_text in plain_words))
                            for block_kind, plain_words in block_words)
        proofread_job = ProofreadJob(
            work_directory=work_directory.resolve(), model_directory=self._model_directory.resolve(),
            device_name=self._device_name, text_blocks=text_blocks, block_checks=tuple(block_checks),
            break_checks=tuple(break_checks), proofread_settings=DEFAULT_SETTINGS, result_file=PROOFREAD_RESULT_FILE)
        job_file = work_directory / PROOFREAD_JOB_FILE
        write_json_document(job_file, proofread_job.to_payload())
        stage_progress.advance_stage(0, PROGRESS_STEPS, "starting proofread worker")
        worker_run = self._worker_launcher.launch_worker(self._worker_environment, job_file,
                                                         ProofreadProgressRelay(stage_progress))
        if worker_run.exit_code != 0:
            raise WorkerFailedError(describe_worker_failure("Proofread worker", worker_run))
        return validate_result(read_json_document(proofread_job.resolve_inside(PROOFREAD_RESULT_FILE)), text_blocks)
