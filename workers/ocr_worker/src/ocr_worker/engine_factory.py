"""Maps a job's backend name to the OCR engine that serves it.

Add a backend by registering a new builder in ``ENGINE_BUILDERS``; no existing
code needs to change.
"""

from collections.abc import Callable

from ocr_worker.engine_protocol import OcrEngine
from worker_protocol.ocr_job_schema import HTTP_BACKEND, TRANSFORMERS_BACKEND, OcrJobSpec


def _build_transformers_engine(ocr_request: OcrJobSpec) -> OcrEngine:
    """Create the in-process transformers engine.

    Args:
        ocr_request: Validated job with ``model_directory`` and ``device_name``.

    Returns:
        A loaded ``TransformersOcrEngine``.
    """
    from ocr_worker.transformers_engine import TransformersOcrEngine  # torch import is slow; load only when used

    return TransformersOcrEngine(ocr_request.model_directory, ocr_request.device_name, ocr_request.prompt_text)


def _build_http_engine(ocr_request: OcrJobSpec) -> OcrEngine:
    """Create the OpenAI-compatible HTTP engine.

    Args:
        ocr_request: Validated job with ``server_url`` and ``server_model``.

    Returns:
        A configured ``HttpOcrEngine``.
    """
    from ocr_worker.http_engine import HttpOcrEngine

    return HttpOcrEngine(ocr_request.server_url, ocr_request.server_model, ocr_request.prompt_text)


ENGINE_BUILDERS: dict[str, Callable[[OcrJobSpec], OcrEngine]] = {
    TRANSFORMERS_BACKEND: _build_transformers_engine,
    HTTP_BACKEND: _build_http_engine,
}


def create_ocr_engine(ocr_request: OcrJobSpec) -> OcrEngine:
    """Build the engine for the job's backend.

    Usage:
        Called once by the worker entry point.

    Args:
        ocr_request: Validated OCR job.

    Returns:
        An object implementing ``OcrEngine``.

    Raises:
        ValueError: If no builder is registered for ``ocr_request.backend_name``.
    """
    engine_builder = ENGINE_BUILDERS.get(ocr_request.backend_name)
    if engine_builder is None:
        raise ValueError(f"No OCR engine registered for backend {ocr_request.backend_name!r}.")
    return engine_builder(ocr_request)
