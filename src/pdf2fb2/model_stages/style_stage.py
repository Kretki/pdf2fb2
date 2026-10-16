"""Text-style stage: delegates word bold/italic/underline/strikeout detection to the textar worker venv.

Runs on the final (cleaned and re-segmented) page images. For every text segment that receives
word styles, the detected words and their attributes are stored in the segment's ``words`` field;
``text_styling.style_markup`` later applies them to the OCR text.
"""

from pathlib import Path

from pdf2fb2.progress.progress_contracts import StageProgress
from pdf2fb2.segments.label_roles import receives_word_styles
from pdf2fb2.worker_runtime.worker_environment import WorkerEnvironment
from pdf2fb2.worker_runtime.worker_launcher import WorkerFailedError, WorkerLauncher, describe_worker_failure
from worker_protocol.json_files import read_json_document, write_json_document
from worker_protocol.path_safety import relative_to_directory
from worker_protocol.progress_events import MODEL_LOADING_EVENT, STYLE_PAGE_EVENT, read_counter_field
from worker_protocol.style_job_schema import StyleJob, StylePage

STYLE_JOB_FILE = "style_job.json"
STYLE_RESULT_FILE = "style_result.json"
WORD_FIELDS = ("emphasis_class", "emphasis_score", "line_class", "line_score")


class StyleProgressRelay:
    """Turns textar worker events into ``detecting text styles k/n`` stage progress.

    Usage:
        Passed as the event handler of the textar worker run.
    """

    def __init__(self, stage_progress: StageProgress, page_count: int) -> None:
        """Bind the relay to a stage.

        Args:
            stage_progress: Progress sink of the styling stage.
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
            self._stage_progress.advance_stage(0, self._page_count, "loading text-style models")
        elif progress_event["event"] == STYLE_PAGE_EVENT:
            page_number = read_counter_field(progress_event, "page")
            if page_number is not None:
                self._stage_progress.advance_stage(page_number - 1, self._page_count,
                                                   f"detecting text styles {page_number}/{self._page_count}")


def styled_segment_indexes(page_record: dict) -> list[int]:
    """Indexes of the segments of a page whose text receives word styles.

    Args:
        page_record: Segmented page record.

    Returns:
        Segment indexes in page order.
    """
    return [segment_index for segment_index, page_segment in enumerate(page_record["segments"])
            if receives_word_styles(page_segment["label"])]


def is_valid_word(word_entry: object) -> bool:
    """Check the shape of one word returned by the worker.

    Args:
        word_entry: Decoded JSON value.

    Returns:
        True for ``{"bbox": [4 numbers], classes 0..3, scores 0..1}``.
    """
    if not isinstance(word_entry, dict) or not isinstance(word_entry.get("bbox"), list):
        return False
    if len(word_entry["bbox"]) != 4 or not all(isinstance(coordinate, int) for coordinate in word_entry["bbox"]):
        return False
    return all(isinstance(word_entry.get(field_name), (int, float)) and 0 <= word_entry[field_name] <= 3
               for field_name in WORD_FIELDS)


def store_page_words(page_record: dict, page_result: object) -> None:
    """Copy the validated words of each region into the matching segment.

    Args:
        page_record: Page record (mutated: ``words`` added to styled segments).
        page_result: Result entry of this page.

    Raises:
        WorkerFailedError: If the entry does not match the requested regions.
    """
    segment_indexes = styled_segment_indexes(page_record)
    region_results = page_result.get("regions") if isinstance(page_result, dict) else None
    if not isinstance(region_results, list) or len(region_results) != len(segment_indexes):
        raise WorkerFailedError(f"Textar worker returned regions that do not match page {page_record['page']}.")
    for segment_index, region_words in zip(segment_indexes, region_results, strict=True):
        if not isinstance(region_words, list) or not all(is_valid_word(word_entry) for word_entry in region_words):
            raise WorkerFailedError(f"Textar worker returned malformed words on page {page_record['page']}.")
        page_record["segments"][segment_index]["words"] = region_words


class StyleStage:
    """Detects word text attributes by running the textar worker in its own venv.

    Usage:
        Build once with injected launcher/environment and call ``detect_text_styles``.
    """

    def __init__(self, worker_launcher: WorkerLauncher, worker_environment: WorkerEnvironment,
                 textar_directory: Path, detector_directory: Path, device_name: str) -> None:
        """Store the stage configuration.

        Args:
            worker_launcher: Starts the worker process.
            worker_environment: Textar worker venv description.
            textar_directory: Directory with ``TexTAR-trained.pt``.
            detector_directory: Directory with the docTR ``db_resnet50`` weights.
            device_name: Device such as ``gpu:0`` or ``cpu``.
        """
        self._worker_launcher = worker_launcher
        self._worker_environment = worker_environment
        self._textar_directory = textar_directory
        self._detector_directory = detector_directory
        self._device_name = device_name

    def detect_text_styles(self, page_records: list[dict], work_directory: Path,
                           stage_progress: StageProgress) -> None:
        """Detect the words and their attributes inside the text segments of all pages.

        Args:
            page_records: Final segmented page records whose images lie inside ``work_directory``
                (mutated: ``words`` added to styled segments).
            work_directory: Output directory for the job and result files.
            stage_progress: Receives ``detecting text styles k/n`` updates.

        Raises:
            WorkerFailedError: If the worker fails or returns a mismatching result.
        """
        style_job = self._build_job(page_records, work_directory)
        job_file = work_directory / STYLE_JOB_FILE
        write_json_document(job_file, style_job.to_payload())
        stage_progress.advance_stage(0, len(page_records), "starting text-style worker")
        worker_run = self._worker_launcher.launch_worker(self._worker_environment, job_file,
                                                         StyleProgressRelay(stage_progress, len(page_records)))
        if worker_run.exit_code != 0:
            raise WorkerFailedError(describe_worker_failure("Textar worker", worker_run))
        page_results = read_json_document(style_job.resolve_inside(STYLE_RESULT_FILE)).get("pages")
        if not isinstance(page_results, list) or len(page_results) != len(page_records):
            raise WorkerFailedError("Textar worker returned a result that does not match the page list.")
        for page_record, page_result in zip(page_records, page_results, strict=True):
            store_page_words(page_record, page_result)

    def _build_job(self, page_records: list[dict], work_directory: Path) -> StyleJob:
        """Create the style job for the given pages.

        Args:
            page_records: Segmented page records.
            work_directory: Output directory.

        Returns:
            StyleJob with paths relative to ``work_directory`` and the text regions of every page.
        """
        style_pages = tuple(
            StylePage(image_file=relative_to_directory(work_directory, Path(page_record["image"])),
                      text_regions=tuple(tuple(float(coordinate) for coordinate in page_record["segments"]
                                               [segment_index]["bbox"])
                                         for segment_index in styled_segment_indexes(page_record)))
            for page_record in page_records)
        return StyleJob(work_directory=work_directory.resolve(), textar_directory=self._textar_directory.resolve(),
                        detector_directory=self._detector_directory.resolve(), device_name=self._device_name,
                        style_pages=style_pages, result_file=STYLE_RESULT_FILE)
