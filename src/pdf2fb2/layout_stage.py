"""Page segmentation stage: delegates PP-DocLayoutV3 to the layout worker venv."""

import math
from pathlib import Path

from PIL import Image

from pdf2fb2.progress_contracts import StageProgress
from pdf2fb2.segment_geometry import order_segments_for_reading, remove_nested_segments
from pdf2fb2.worker_environment import WorkerEnvironment
from pdf2fb2.worker_launcher import WorkerFailedError, WorkerLauncher, describe_worker_failure
from worker_protocol.job_schema import LayoutJob
from worker_protocol.json_files import read_json_document, write_json_document
from worker_protocol.path_safety import relative_to_directory
from worker_protocol.progress_events import LAYOUT_PAGE_EVENT, MODEL_LOADING_EVENT, read_counter_field

LAYOUT_JOB_FILE = "layout_job.json"
LAYOUT_RESULT_FILE = "layout_result.json"


def _parse_box(raw_layout_box: object) -> dict | None:
    """Validate one box from the worker result.

    Args:
        raw_layout_box: Decoded JSON value from the layout result.

    Returns:
        Segment dict ``{"label", "bbox", "score", "order"}``, or None if the box is malformed.
    """
    if not isinstance(raw_layout_box, dict) or not isinstance(raw_layout_box.get("label"), str):
        return None
    raw_coordinates = raw_layout_box.get("bbox")
    if not isinstance(raw_coordinates, list) or len(raw_coordinates) != 4:
        return None
    if not all(isinstance(coordinate, (int, float)) and math.isfinite(coordinate) for coordinate in raw_coordinates):
        return None
    raw_order = raw_layout_box.get("order")
    return {"label": raw_layout_box["label"][:64], "bbox": [float(coordinate) for coordinate in raw_coordinates],
            "score": float(raw_layout_box.get("score") or 0.0),
            "order": raw_order if isinstance(raw_order, int) else None}


def build_page_record(page_number: int, image_path: Path, raw_boxes: list) -> dict:
    """Turn the worker's boxes for one page into a cleaned, ordered page record.

    Usage:
        Called by ``LayoutStage`` for every page of the result.

    Args:
        page_number: 1-based page number.
        image_path: Rendered page image.
        raw_boxes: Boxes from the layout worker for this page.

    Returns:
        Page record ``{"page", "image", "size", "segments"}``.
    """
    with Image.open(image_path) as page_image:
        page_width, page_height = page_image.size
    parsed_boxes = [parsed_box for parsed_box in map(_parse_box, raw_boxes) if parsed_box is not None]
    page_segments = order_segments_for_reading(remove_nested_segments(parsed_boxes), page_width)
    return {"page": page_number, "image": str(image_path), "size": [page_width, page_height],
            "segments": page_segments}


class LayoutProgressRelay:
    """Turns layout worker events into ``segmenting page k/n`` stage progress.

    Usage:
        Passed as the event handler of the layout worker run.
    """

    def __init__(self, stage_progress: StageProgress, page_count: int) -> None:
        """Bind the relay to a stage.

        Args:
            stage_progress: Progress sink of the segmentation stage.
            page_count: Number of pages in the job (> 0).
        """
        self._stage_progress = stage_progress
        self._page_count = page_count
        self._event_handlers = {MODEL_LOADING_EVENT: self._show_model_loading, LAYOUT_PAGE_EVENT: self._show_page}

    def __call__(self, progress_event: dict) -> None:
        """Handle one event; unknown events are ignored.

        Args:
            progress_event: Event decoded by the launcher.

        Returns:
            None.
        """
        event_handler = self._event_handlers.get(progress_event["event"])
        if event_handler is not None:
            event_handler(progress_event)

    def _show_model_loading(self, progress_event: dict) -> None:
        """Report that PP-DocLayoutV3 is being loaded.

        Args:
            progress_event: Model-loading event (no fields used).
        """
        self._stage_progress.advance_stage(0, self._page_count, "loading layout model")

    def _show_page(self, progress_event: dict) -> None:
        """Report the page the worker has started on.

        Args:
            progress_event: Layout-page event with the 1-based ``page``.
        """
        page_number = read_counter_field(progress_event, "page")
        if page_number is not None:
            self._stage_progress.advance_stage(page_number - 1, self._page_count,
                                               f"segmenting page {page_number}/{self._page_count}")


class LayoutStage:
    """Segments rendered pages by running the layout worker in its own venv.

    Usage:
        Build once with injected launcher/environment and call ``segment_pages``.
    """

    def __init__(self, worker_launcher: WorkerLauncher, worker_environment: WorkerEnvironment,
                 model_directory: Path, device_name: str, score_threshold: float) -> None:
        """Store the stage configuration.

        Args:
            worker_launcher: Starts the worker process.
            worker_environment: Layout worker venv description.
            model_directory: Directory with the PP-DocLayoutV3 weights.
            device_name: Paddle device such as ``gpu:0`` or ``cpu``.
            score_threshold: Minimum detection score, 0..1.
        """
        self._worker_launcher = worker_launcher
        self._worker_environment = worker_environment
        self._model_directory = model_directory
        self._device_name = device_name
        self._score_threshold = score_threshold

    def segment_pages(self, page_images: list[Path], work_directory: Path,
                      stage_progress: StageProgress) -> list[dict]:
        """Segment all pages with PP-DocLayoutV3.

        Args:
            page_images: Rendered page images inside ``work_directory``, in page order.
            work_directory: Output directory for ``layout_job.json`` / ``layout_result.json``.
            stage_progress: Receives ``segmenting page k/n`` updates.

        Returns:
            One page record per image, in page order.

        Raises:
            WorkerFailedError: If the worker fails or returns a mismatching result.
        """
        layout_job = self._build_job(page_images, work_directory)
        job_file = work_directory / LAYOUT_JOB_FILE
        write_json_document(job_file, layout_job.to_payload())
        stage_progress.advance_stage(0, len(page_images), "starting layout worker")
        worker_run = self._worker_launcher.launch_worker(self._worker_environment, job_file,
                                                         LayoutProgressRelay(stage_progress, len(page_images)))
        if worker_run.exit_code != 0:
            raise WorkerFailedError(describe_worker_failure("Layout worker", worker_run))
        page_results = read_json_document(layout_job.resolve_inside(LAYOUT_RESULT_FILE)).get("pages")
        if not isinstance(page_results, list) or len(page_results) != len(page_images) \
                or not all(isinstance(page_result, dict) for page_result in page_results):
            raise WorkerFailedError("Layout worker returned a result that does not match the page list.")
        paired_results = zip(page_images, page_results, strict=True)
        return [build_page_record(page_number, image_path, page_result.get("boxes") or [])
                for page_number, (image_path, page_result) in enumerate(paired_results, start=1)]

    def _build_job(self, page_images: list[Path], work_directory: Path) -> LayoutJob:
        """Create the layout job for the given pages.

        Args:
            page_images: Rendered page images inside ``work_directory``.
            work_directory: Output directory.

        Returns:
            LayoutJob with paths relative to ``work_directory``.
        """
        return LayoutJob(
            work_directory=work_directory.resolve(),
            model_directory=self._model_directory.resolve(),
            device_name=self._device_name,
            score_threshold=self._score_threshold,
            page_images=tuple(relative_to_directory(work_directory, image_path) for image_path in page_images),
            result_file=LAYOUT_RESULT_FILE,
        )
