"""Reader-mark cleaning stage: delegates handwriting/pen-mark removal to the readermarks worker venv.

Runs after layout and before OCR. Picture segments found by the layout model are sent as masked regions,
so the worker blanks them before detection (printed pictures are never taken for hand drawings).
Only pages with reader marks are returned, as cleaned images for a rescan.
The same worker also inspects single page regions (everything around the region blanked) for the
handwriting check of position outliers.
"""

import shutil
from pathlib import Path

from pdf2fb2.pipeline.stage_contracts import IsolatedRegion
from pdf2fb2.progress.progress_contracts import StageProgress
from pdf2fb2.segments.label_roles import PICTURE_LABELS
from pdf2fb2.segments.segment_geometry import surrounding_boxes
from pdf2fb2.worker_runtime.worker_environment import WorkerEnvironment
from pdf2fb2.worker_runtime.worker_launcher import WorkerFailedError, WorkerLauncher, describe_worker_failure
from worker_protocol.json_files import read_json_document, write_json_document
from worker_protocol.marks_job_schema import MarksJob, RegionBox
from worker_protocol.path_safety import relative_to_directory
from worker_protocol.progress_events import MARKS_PAGE_EVENT, MODEL_LOADING_EVENT, read_counter_field

MARKS_JOB_FILE = "marks_job.json"
MARKS_RESULT_FILE = "marks_result.json"
MARKS_DIRECTORY = "marks"
REGIONS_DIRECTORY = "regions"
CLEANING_STEP_TEXT = "cleaning reader marks"
REGION_STEP_TEXT = "checking outlying segment"


class MarksProgressRelay:
    """Turns readermarks worker events into ``<step text> k/n`` stage progress.

    Usage:
        Passed as the event handler of the readermarks worker run.
    """

    def __init__(self, stage_progress: StageProgress, page_count: int, step_text: str = CLEANING_STEP_TEXT) -> None:
        """Bind the relay to a stage.

        Args:
            stage_progress: Progress sink of the running stage.
            page_count: Number of images in the job (> 0).
            step_text: Description of one image's work, e.g. ``cleaning reader marks``.
        """
        self._stage_progress = stage_progress
        self._page_count = page_count
        self._step_text = step_text

    def announce_start(self) -> None:
        """Report that the worker process is being started.

        Returns:
            None.
        """
        self._stage_progress.advance_stage(0, self._page_count, "starting reader-mark worker")

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
                                                   f"{self._step_text} {page_number}/{self._page_count}")


def picture_regions(page_record: dict) -> tuple[RegionBox, ...]:
    """Boxes of the picture-like segments of a page record.

    Args:
        page_record: Segmented page record.

    Returns:
        ``(x_start, y_start, x_end, y_end)`` pixel boxes.
    """
    return tuple(tuple(float(coordinate) for coordinate in page_segment["bbox"])
                 for page_segment in page_record["segments"] if page_segment["label"] in PICTURE_LABELS)


def isolation_masks(isolated_region: IsolatedRegion) -> tuple[RegionBox, ...]:
    """Boxes that blank the whole page except the inspected region.

    Args:
        isolated_region: Region to inspect alone.

    Returns:
        Up to four ``(x_start, y_start, x_end, y_end)`` pixel boxes around the region.
    """
    return tuple(tuple(surrounding_box)
                 for surrounding_box in surrounding_boxes(isolated_region.region_box, isolated_region.page_size))


def stage_region_images(isolated_regions: list[IsolatedRegion], regions_directory: Path) -> list[Path]:
    """Copy the page of every region under a unique name (the worker keys its outputs by file stem).

    Args:
        isolated_regions: Regions to inspect, possibly several per page.
        regions_directory: Directory receiving ``region_NNNN<suffix>`` copies (created if missing).

    Returns:
        Copied images in region order.
    """
    regions_directory.mkdir(parents=True, exist_ok=True)
    staged_images = []
    for region_index, isolated_region in enumerate(isolated_regions, start=1):
        staged_image = regions_directory / f"region_{region_index:04d}{isolated_region.page_image.suffix}"
        shutil.copyfile(isolated_region.page_image, staged_image)
        staged_images.append(staged_image)
    return staged_images


def validate_page_results(page_results: object, expected_count: int) -> list[dict]:
    """Check that the worker returned one result object per job image.

    Args:
        page_results: ``pages`` value of the result document.
        expected_count: Number of images in the job.

    Returns:
        The per-image results, in job order.

    Raises:
        WorkerFailedError: If the result does not match the job.
    """
    if not isinstance(page_results, list) or len(page_results) != expected_count \
            or not all(isinstance(page_result, dict) for page_result in page_results):
        raise WorkerFailedError("Readermarks worker returned a result that does not match the page list.")
    return page_results


def collect_cleaned_pages(marks_job: MarksJob, page_results: list[dict], page_records: list[dict]) -> dict[int, Path]:
    """Pick the cleaned images of pages with marks.

    Args:
        marks_job: Job that was run.
        page_results: Validated per-page results, in job order.
        page_records: Page records of the job, in job order.

    Returns:
        Mapping page number -> cleaned image, only for pages with reader marks.

    Raises:
        ValueError: If a cleaned image path escapes the work directory.
    """
    cleaned_pages = {}
    for page_record, page_result in zip(page_records, page_results, strict=True):
        if page_result.get("has_marks") is True and isinstance(page_result.get("clean_file"), str):
            cleaned_pages[page_record["page"]] = marks_job.resolve_inside(page_result["clean_file"])
    return cleaned_pages


class MarksStage:
    """Runs the readermarks worker in its own venv: cleans whole pages or inspects single regions.

    Usage:
        Build once with injected launcher/environment and call ``clean_pages`` or ``find_marked_regions``.
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
        marks_job, page_results = self._run_job(
            [Path(page_record["image"]) for page_record in page_records],
            tuple(picture_regions(page_record) for page_record in page_records),
            work_directory, MarksProgressRelay(stage_progress, len(page_records)))
        return collect_cleaned_pages(marks_job, page_results, page_records)

    def find_marked_regions(self, isolated_regions: list[IsolatedRegion], work_directory: Path,
                            stage_progress: StageProgress) -> set[int]:
        """Detect reader marks inside every region, with the rest of its page blanked.

        Args:
            isolated_regions: Regions to inspect (at least one).
            work_directory: Directory for staged page copies, job, result and ``marks/`` outputs.
            stage_progress: Receives ``checking outlying segment k/n`` updates.

        Returns:
            Indexes into ``isolated_regions`` of the regions with reader marks.

        Raises:
            WorkerFailedError: If the worker fails or returns a mismatching result.
        """
        staged_images = stage_region_images(isolated_regions, work_directory / REGIONS_DIRECTORY)
        _, page_results = self._run_job(
            staged_images, tuple(isolation_masks(isolated_region) for isolated_region in isolated_regions),
            work_directory, MarksProgressRelay(stage_progress, len(isolated_regions), REGION_STEP_TEXT))
        return {region_index for region_index, page_result in enumerate(page_results)
                if page_result.get("has_marks") is True}

    def _run_job(self, page_images: list[Path], masked_regions: tuple[tuple[RegionBox, ...], ...],
                 work_directory: Path, progress_relay: MarksProgressRelay) -> tuple[MarksJob, list[dict]]:
        """Write the job, run the worker and read its validated per-image results.

        Args:
            page_images: Images inside ``work_directory``, in job order.
            masked_regions: Per image, boxes blanked before detection.
            work_directory: Output directory for the job, result and ``marks/`` outputs.
            progress_relay: Event handler of the worker run.

        Returns:
            The job that was run and its per-image results, in job order.

        Raises:
            WorkerFailedError: If the worker fails or returns a mismatching result.
        """
        marks_job = self._build_job(page_images, masked_regions, work_directory)
        (work_directory / MARKS_DIRECTORY).mkdir(parents=True, exist_ok=True)
        job_file = work_directory / MARKS_JOB_FILE
        write_json_document(job_file, marks_job.to_payload())
        progress_relay.announce_start()
        worker_run = self._worker_launcher.launch_worker(self._worker_environment, job_file, progress_relay)
        if worker_run.exit_code != 0:
            raise WorkerFailedError(describe_worker_failure("Readermarks worker", worker_run))
        result_document = read_json_document(marks_job.resolve_inside(MARKS_RESULT_FILE))
        return marks_job, validate_page_results(result_document.get("pages"), len(page_images))

    def _build_job(self, page_images: list[Path], masked_regions: tuple[tuple[RegionBox, ...], ...],
                   work_directory: Path) -> MarksJob:
        """Create the marks job for the given images.

        Args:
            page_images: Images inside ``work_directory``.
            masked_regions: Per image, boxes blanked before detection.
            work_directory: Output directory.

        Returns:
            MarksJob with paths relative to ``work_directory``.
        """
        return MarksJob(
            work_directory=work_directory.resolve(), checkpoint_directory=self._checkpoint_directory.resolve(),
            device_name=self._device_name,
            page_images=tuple(relative_to_directory(work_directory, page_image) for page_image in page_images),
            output_directory=MARKS_DIRECTORY, result_file=MARKS_RESULT_FILE, write_debug=self._write_debug,
            masked_regions=masked_regions)
