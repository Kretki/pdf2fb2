"""Reader-mark cleaning stage: delegates handwriting/pen-mark removal to the readermarks worker venv.

Runs after layout and before OCR. Picture segments found by the layout model are sent as masked regions,
so the worker blanks them before detection (printed pictures are never taken for hand drawings).
Only pages with reader marks are returned, as cleaned images for a rescan.
"""

from pathlib import Path

from pdf2fb2.progress.progress_contracts import StageProgress
from pdf2fb2.segments.label_roles import PICTURE_LABELS
from pdf2fb2.worker_runtime.worker_environment import WorkerEnvironment
from pdf2fb2.worker_runtime.worker_launcher import WorkerFailedError, WorkerLauncher, describe_worker_failure
from worker_protocol.json_files import read_json_document, write_json_document
from worker_protocol.marks_job_schema import MarksJob
from worker_protocol.path_safety import relative_to_directory
from worker_protocol.progress_events import MARKS_PAGE_EVENT, MODEL_LOADING_EVENT, read_counter_field

MARKS_JOB_FILE = "marks_job.json"
MARKS_RESULT_FILE = "marks_result.json"
MARKS_DIRECTORY = "marks"


class MarksProgressRelay:
    """Turns readermarks worker events into ``cleaning reader marks k/n`` stage progress.

    Usage:
        Passed as the event handler of the readermarks worker run.
    """

    def __init__(self, stage_progress: StageProgress, page_count: int) -> None:
        """Bind the relay to a stage.

        Args:
            stage_progress: Progress sink of the cleaning stage.
            page_count: Number of pages in the job (> 0).
        """
        self._stage_progress = stage_progress
        self._page_count = page_count

    def __call__(self, progress_event: dict) -> None:
        """Handle one event; unknown events are ignored.

        Args:
            progress_event: Event decoded by the launcher.

        Returns:
            None.
        """
        if progress_event["event"] == MODEL_LOADING_EVENT:
            self._stage_progress.advance_stage(0, self._page_count, "loading reader-mark models")
        elif progress_event["event"] == MARKS_PAGE_EVENT:
            page_number = read_counter_field(progress_event, "page")
            if page_number is not None:
                self._stage_progress.advance_stage(page_number - 1, self._page_count,
                                                   f"cleaning reader marks {page_number}/{self._page_count}")


def picture_regions(page_record: dict) -> tuple[tuple[float, float, float, float], ...]:
    """Boxes of the picture-like segments of a page record.

    Args:
        page_record: Segmented page record.

    Returns:
        ``(x_start, y_start, x_end, y_end)`` pixel boxes.
    """
    return tuple(tuple(float(coordinate) for coordinate in page_segment["bbox"])
                 for page_segment in page_record["segments"] if page_segment["label"] in PICTURE_LABELS)


def collect_cleaned_pages(marks_job: MarksJob, page_results: object, page_records: list[dict]) -> dict[int, Path]:
    """Validate the worker result and pick the cleaned images of pages with marks.

    Args:
        marks_job: Job that was run.
        page_results: ``pages`` value of the result document.
        page_records: Page records of the job, in job order.

    Returns:
        Mapping page number -> cleaned image, only for pages with reader marks.

    Raises:
        WorkerFailedError: If the result does not match the job.
    """
    if not isinstance(page_results, list) or len(page_results) != len(page_records) \
            or not all(isinstance(page_result, dict) for page_result in page_results):
        raise WorkerFailedError("Readermarks worker returned a result that does not match the page list.")
    cleaned_pages = {}
    for page_record, page_result in zip(page_records, page_results, strict=True):
        if page_result.get("has_marks") is True and isinstance(page_result.get("clean_file"), str):
            cleaned_pages[page_record["page"]] = marks_job.resolve_inside(page_result["clean_file"])
    return cleaned_pages


class MarksStage:
    """Removes reader marks from rendered pages by running the readermarks worker in its own venv.

    Usage:
        Build once with injected launcher/environment and call ``clean_pages``.
    """

    def __init__(self, worker_launcher: WorkerLauncher, worker_environment: WorkerEnvironment,
                 checkpoint_directory: Path, device_name: str, write_debug: bool) -> None:
        """Store the stage configuration.

        Args:
            worker_launcher: Starts the worker process.
            worker_environment: Readermarks worker venv description.
            checkpoint_directory: Directory with ``spotter.pt``/``segmenter.pt``/``thresholds.json``.
            device_name: Device such as ``gpu:0`` or ``cpu``.
            write_debug: Whether the worker writes debug overlays.
        """
        self._worker_launcher = worker_launcher
        self._worker_environment = worker_environment
        self._checkpoint_directory = checkpoint_directory
        self._device_name = device_name
        self._write_debug = write_debug

    def clean_pages(self, page_records: list[dict], work_directory: Path,
                    stage_progress: StageProgress) -> dict[int, Path]:
        """Detect and remove reader marks on all pages, with picture segments blanked.

        Args:
            page_records: Segmented page records whose images lie inside ``work_directory``.
            work_directory: Output directory for the job, result and ``marks/`` outputs.
            stage_progress: Receives ``cleaning reader marks k/n`` updates.

        Returns:
            Mapping page number -> cleaned image, only for pages with reader marks.

        Raises:
            WorkerFailedError: If the worker fails or returns a mismatching result.
        """
        marks_job = self._build_job(page_records, work_directory)
        (work_directory / MARKS_DIRECTORY).mkdir(parents=True, exist_ok=True)
        job_file = work_directory / MARKS_JOB_FILE
        write_json_document(job_file, marks_job.to_payload())
        stage_progress.advance_stage(0, len(page_records), "starting reader-mark worker")
        worker_run = self._worker_launcher.launch_worker(self._worker_environment, job_file,
                                                         MarksProgressRelay(stage_progress, len(page_records)))
        if worker_run.exit_code != 0:
            raise WorkerFailedError(describe_worker_failure("Readermarks worker", worker_run))
        result_document = read_json_document(marks_job.resolve_inside(MARKS_RESULT_FILE))
        return collect_cleaned_pages(marks_job, result_document.get("pages"), page_records)

    def _build_job(self, page_records: list[dict], work_directory: Path) -> MarksJob:
        """Create the marks job for the given pages.

        Args:
            page_records: Segmented page records.
            work_directory: Output directory.

        Returns:
            MarksJob with paths relative to ``work_directory`` and the picture boxes of every page.
        """
        return MarksJob(
            work_directory=work_directory.resolve(), checkpoint_directory=self._checkpoint_directory.resolve(),
            device_name=self._device_name,
            page_images=tuple(relative_to_directory(work_directory, Path(page_record["image"]))
                              for page_record in page_records),
            output_directory=MARKS_DIRECTORY, result_file=MARKS_RESULT_FILE, write_debug=self._write_debug,
            masked_regions=tuple(picture_regions(page_record) for page_record in page_records))
