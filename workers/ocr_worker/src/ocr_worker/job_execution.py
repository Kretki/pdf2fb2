"""Runs an OCR job against any engine that follows ``OcrEngine``."""

from concurrent.futures import ThreadPoolExecutor
from functools import partial

from PIL import Image

from ocr_worker.engine_protocol import OcrEngine
from worker_protocol.json_files import append_json_line
from worker_protocol.ocr_job_schema import OcrCropItem, OcrJobSpec
from worker_protocol.progress_events import OCR_PROGRESS_EVENT, emit_progress_event


def recognize_item(ocr_request: OcrJobSpec, ocr_engine: OcrEngine, ocr_item: OcrCropItem) -> dict:
    """Read one crop, converting any failure into an error record.

    Usage:
        Mapped over all job items by ``run_ocr_job``; one bad crop must not abort the job.

    Args:
        ocr_request: Job the item belongs to (resolves the crop path).
        ocr_engine: Engine that reads the crop.
        ocr_item: Crop to read.

    Returns:
        Result record ``{"item_key": str, "text": str, "error": str | None}``.
    """
    try:
        with Image.open(ocr_request.resolve_inside(ocr_item.image_file)) as crop_file:
            crop_image = crop_file.convert("RGB")
        recognized_text = ocr_engine.recognize_image(crop_image, ocr_item.max_new_tokens)
        return {"item_key": ocr_item.item_key, "text": recognized_text, "error": None}
    except Exception as recognition_error:  # noqa: BLE001 - reported per item, the job keeps going
        return {"item_key": ocr_item.item_key, "text": "",
                "error": f"{type(recognition_error).__name__}: {recognition_error}"}


def run_ocr_job(ocr_request: OcrJobSpec, ocr_engine: OcrEngine) -> int:
    """Read every crop of a job and append the results to the job's result file.

    Usage:
        Called by the worker entry point. Results are appended line by line, so
        progress survives a crash. Emits an ``ocr_progress`` event before the first
        crop and after every finished crop (results arrive in job order).

    Args:
        ocr_request: Validated OCR job.
        ocr_engine: Engine used for every crop; parallelism is used only when
            the engine supports it.

    Returns:
        Number of crops that failed.
    """
    result_path = ocr_request.resolve_inside(ocr_request.result_file)
    thread_count = ocr_request.parallel_requests if ocr_engine.supports_parallel_requests else 1
    failure_count = 0
    item_reader = partial(recognize_item, ocr_request, ocr_engine)
    item_count = len(ocr_request.ocr_items)
    emit_progress_event(OCR_PROGRESS_EVENT, done=0, total=item_count)
    with ThreadPoolExecutor(max_workers=thread_count) as thread_pool:
        for finished_count, result_record in enumerate(thread_pool.map(item_reader, ocr_request.ocr_items), start=1):
            append_json_line(result_path, result_record)
            failure_count += result_record["error"] is not None
            emit_progress_event(OCR_PROGRESS_EVENT, done=finished_count, total=item_count)
    return failure_count
