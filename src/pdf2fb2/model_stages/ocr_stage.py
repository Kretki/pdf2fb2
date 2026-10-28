"""OCR stage: crops segments and delegates Qianfan-OCR to the OCR worker venv."""

from dataclasses import dataclass
from pathlib import Path

from pdf2fb2.fb2_output.text_markup import clean_ocr_text, normalize_ellipsis
from pdf2fb2.model_stages.ocr_confidence_cache import CONFIDENCE_CACHE_FILE, OcrConfidenceCache
from pdf2fb2.model_stages.ocr_crop_writer import OcrCropWriter, PendingCrop
from pdf2fb2.model_stages.ocr_text_cache import OcrTextCache, build_segment_key
from pdf2fb2.progress.progress_contracts import StageProgress
from pdf2fb2.segments.label_roles import QIANFAN_PROMPT, requires_ocr
from pdf2fb2.worker_runtime.worker_environment import WorkerEnvironment
from pdf2fb2.worker_runtime.worker_launcher import WorkerFailedError, WorkerLauncher, describe_worker_failure
from worker_protocol.json_files import read_json_lines, write_json_document
from worker_protocol.ocr_job_schema import OcrCropItem, OcrJobSpec
from worker_protocol.progress_events import MODEL_LOADING_EVENT, OCR_PROGRESS_EVENT, read_counter_field

OCR_JOB_FILE = "ocr_job.json"
OCR_RESULT_FILE = "ocr_results.jsonl"
OCR_CACHE_FILE = "ocr_cache.json"
ACCEPTED_EXIT_CODES = frozenset({0, 1})  # 1 = some crops failed; they are reported and retried next run


@dataclass(frozen=True)
class OcrBackendSettings:
    """How the OCR worker should run Qianfan-OCR.

    Attributes:
        backend_name: ``transformers`` (local weights) or ``http`` (OpenAI-compatible server).
        model_directory: Qianfan-OCR weights directory (transformers backend).
        device_name: Device such as ``gpu:0`` (transformers backend).
        server_url: Server base URL (http backend).
        server_model: Model name exposed by the server (http backend).
        parallel_requests: Concurrent requests, 1..64 (http backend).
        prompt_text: Instruction sent with every crop.
        collect_confidences: Ask the worker for per-character log-probabilities (verifier evidence).
    """

    backend_name: str
    model_directory: Path | None
    device_name: str
    server_url: str | None
    server_model: str | None
    parallel_requests: int
    prompt_text: str = QIANFAN_PROMPT
    collect_confidences: bool = True


def apply_cached_texts(page_records: list[dict], text_cache: OcrTextCache,
                       confidence_cache: OcrConfidenceCache | None = None) -> None:
    """Copy cached OCR texts (and confidences, when cached) into the segments, typeset ellipses normalised.

    Args:
        page_records: Page records (segments mutated: ``text`` and, when known, ``ocr_confidence`` are set).
        text_cache: Cache with OCR results.
        confidence_cache: Cache with raw outputs and per-character log-probabilities, or None.

    Returns:
        None.
    """
    for page_record in page_records:
        for page_segment in page_record["segments"]:
            if not requires_ocr(page_segment["label"]):
                continue
            segment_key = build_segment_key(page_record, page_segment)
            page_segment["text"] = normalize_ellipsis(text_cache.lookup_text(segment_key))
            confidence_entry = None if confidence_cache is None else confidence_cache.lookup_entry(segment_key)
            if confidence_entry is not None:
                page_segment["ocr_confidence"] = confidence_entry


class OcrProgressRelay:
    """Turns OCR worker events into ``parsing block b/B on page p/P`` stage progress.

    Usage:
        Passed as the event handler of the OCR worker run.
    """

    def __init__(self, stage_progress: StageProgress, pending_crops: list[PendingCrop]) -> None:
        """Bind the relay to a stage.

        Args:
            stage_progress: Progress sink of the recognition stage.
            pending_crops: Crops in job order (the worker reports results in this order).
        """
        self._stage_progress = stage_progress
        self._pending_crops = pending_crops
        self._event_handlers = {MODEL_LOADING_EVENT: self._show_model_loading, OCR_PROGRESS_EVENT: self._show_crop}

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
        """Report that Qianfan-OCR is being loaded.

        Args:
            progress_event: Model-loading event (no fields used).
        """
        self._stage_progress.advance_stage(0, len(self._pending_crops), "loading OCR model")

    def _show_crop(self, progress_event: dict) -> None:
        """Report finished crops and name the crop being read next.

        Args:
            progress_event: OCR-progress event with ``done`` (finished crops so far).
        """
        finished_count = read_counter_field(progress_event, "done")
        if finished_count is None:
            return
        crop_count = len(self._pending_crops)
        status_text = (f"parsing {self._pending_crops[finished_count].block_label}" if finished_count < crop_count
                       else "OCR finished")
        self._stage_progress.advance_stage(finished_count, crop_count, status_text)


class OcrStage:
    """Reads segment texts by running the OCR worker in its own venv.

    Usage:
        Build once with injected launcher/environment/settings and call ``recognize_segments``.
    """

    def __init__(self, worker_launcher: WorkerLauncher, worker_environment: WorkerEnvironment,
                 backend_settings: OcrBackendSettings) -> None:
        """Store the stage configuration.

        Args:
            worker_launcher: Starts the worker process.
            worker_environment: OCR worker venv description.
            backend_settings: How Qianfan-OCR should be run.
        """
        self._worker_launcher = worker_launcher
        self._worker_environment = worker_environment
        self._backend_settings = backend_settings

    def recognize_segments(self, page_records: list[dict], work_directory: Path,
                           stage_progress: StageProgress) -> None:
        """Read all OCR-able segments, reusing the cache from earlier runs.

        Args:
            page_records: Page records (segments mutated: ``text`` is set).
            work_directory: Output directory for crops, cache, job and result files.
            stage_progress: Receives ``parsing block b/B on page p/P`` updates and failure messages.

        Returns:
            None.

        Raises:
            WorkerFailedError: If the worker crashes (partial results are still cached).
        """
        text_cache = OcrTextCache(work_directory / OCR_CACHE_FILE)
        confidence_cache = OcrConfidenceCache(work_directory / CONFIDENCE_CACHE_FILE)
        crop_writer = OcrCropWriter(work_directory, text_cache, len(page_records))
        pending_crops = crop_writer.write_pending_crops(page_records, stage_progress)
        if pending_crops:
            self._run_worker(pending_crops, work_directory, (text_cache, confidence_cache), stage_progress)
        apply_cached_texts(page_records, text_cache, confidence_cache)

    def _run_worker(self, pending_crops: list[PendingCrop], work_directory: Path,
                    result_caches: tuple[OcrTextCache, OcrConfidenceCache], stage_progress: StageProgress) -> None:
        """Write the job, run the worker and merge its results into the cache.

        Args:
            pending_crops: Crops to read, in job order.
            work_directory: Output directory.
            result_caches: Text and confidence caches that receive the results (saved even if the worker fails).
            stage_progress: Receives per-crop progress and failure messages.

        Returns:
            None.

        Raises:
            WorkerFailedError: If the worker exits with an unexpected code.
        """
        ocr_request = self._build_job([pending_crop.ocr_item for pending_crop in pending_crops], work_directory)
        result_path = ocr_request.resolve_inside(OCR_RESULT_FILE)
        result_path.unlink(missing_ok=True)
        job_file = work_directory / OCR_JOB_FILE
        write_json_document(job_file, ocr_request.to_payload())
        stage_progress.advance_stage(0, len(pending_crops), "starting OCR worker")
        try:
            worker_run = self._worker_launcher.launch_worker(self._worker_environment, job_file,
                                                             OcrProgressRelay(stage_progress, pending_crops))
        finally:
            _merge_results(result_path, result_caches, stage_progress)
        if worker_run.exit_code not in ACCEPTED_EXIT_CODES:
            raise WorkerFailedError(describe_worker_failure("OCR worker", worker_run))

    def _build_job(self, pending_items: list[OcrCropItem], work_directory: Path) -> OcrJobSpec:
        """Create and validate the OCR job.

        Args:
            pending_items: Crops to read.
            work_directory: Output directory.

        Returns:
            Validated OcrJobSpec.

        Raises:
            ValueError: If the backend settings are incomplete.
        """
        backend_settings = self._backend_settings
        ocr_request = OcrJobSpec(
            work_directory=work_directory.resolve(),
            backend_name=backend_settings.backend_name,
            model_directory=None if backend_settings.model_directory is None
            else backend_settings.model_directory.resolve(),
            device_name=backend_settings.device_name,
            server_url=backend_settings.server_url,
            server_model=backend_settings.server_model,
            parallel_requests=backend_settings.parallel_requests,
            prompt_text=backend_settings.prompt_text,
            ocr_items=tuple(pending_items),
            result_file=OCR_RESULT_FILE,
            return_confidences=backend_settings.collect_confidences,
        )
        ocr_request.validate_backend_fields()
        return ocr_request


def _merge_results(result_path: Path, result_caches: tuple[OcrTextCache, OcrConfidenceCache],
                   stage_progress: StageProgress) -> None:
    """Move worker results into the caches and persist them; report failed crops.

    Args:
        result_path: JSON-lines result file written by the worker.
        result_caches: Text cache and confidence cache that receive successful results.
        stage_progress: Receives one message per failed crop.

    Returns:
        None. Failed crops are not cached, so the next run retries them.
    """
    text_cache, confidence_cache = result_caches
    for result_record in read_json_lines(result_path):
        item_key, item_text = result_record.get("item_key"), result_record.get("text")
        if not isinstance(item_key, str):
            continue
        if result_record.get("error"):
            stage_progress.report_message(f"! OCR failed for segment {item_key}: {result_record['error']}")
        elif isinstance(item_text, str):
            text_cache.store_text(item_key, clean_ocr_text(item_text))
            if isinstance(result_record.get("char_logprobs"), list):
                confidence_cache.store_entry(item_key, item_text, result_record["char_logprobs"])
    text_cache.save_to_disk()
    confidence_cache.save_to_disk()
