"""OCR job description (orchestrator -> ocr_worker): segment crops to read.

Result: JSON-lines file, one ``{"item_key": str, "text": str, "error": str | None}`` per crop, plus
``"char_logprobs": [float, ...]`` (one log-probability per character of ``text``) when ``return_confidences`` is set.
All image and result paths are relative to ``work_directory``.
"""

from dataclasses import dataclass
from pathlib import Path

from worker_protocol.field_validation import (
    optional_string,
    require_absolute_directory,
    require_integer,
    require_list,
    require_string,
    validate_device_name,
    validate_http_url,
)
from worker_protocol.job_schema import checked_relative_path, optional_directory
from worker_protocol.path_safety import resolve_inside_directory

MAX_OCR_ITEMS = 2_000_000
MAX_NEW_TOKENS_LIMIT = 32_768
MAX_PARALLEL_REQUESTS = 64
MAX_PROMPT_LENGTH = 8192
TRANSFORMERS_BACKEND = "transformers"
HTTP_BACKEND = "http"
OCR_BACKEND_NAMES = frozenset({TRANSFORMERS_BACKEND, HTTP_BACKEND})


@dataclass(frozen=True)
class OcrCropItem:
    """One segment crop to be read by the OCR model.

    Attributes:
        item_key: Opaque key the orchestrator uses to map the text back to its segment.
        image_file: Crop image path relative to the job's working directory.
        max_new_tokens: Generation budget for this crop, 1..32768.
    """

    item_key: str
    image_file: str
    max_new_tokens: int

    def to_payload(self) -> dict:
        """Serialise the item to a JSON-compatible object.

        Usage:
            Called by ``OcrJobSpec.to_payload``.

        Returns:
            Dictionary with the three item fields.
        """
        return {"item_key": self.item_key, "image_file": self.image_file, "max_new_tokens": self.max_new_tokens}

    @classmethod
    def from_payload(cls, item_payload: dict, work_directory: Path) -> "OcrCropItem":
        """Decode and validate one OCR item.

        Usage:
            Called by ``OcrJobSpec.from_payload`` for every item.

        Args:
            item_payload: Decoded JSON object of the item.
            work_directory: Working directory the image path must stay inside.

        Returns:
            The validated OcrCropItem.

        Raises:
            ValueError: If a field is invalid or the image path escapes ``work_directory``.
        """
        if not isinstance(item_payload, dict):
            raise ValueError("Every OCR item must be a JSON object.")
        return cls(
            item_key=require_string(item_payload, "item_key", 256),
            image_file=checked_relative_path(work_directory, require_string(item_payload, "image_file")),
            max_new_tokens=require_integer(item_payload, "max_new_tokens", 1, MAX_NEW_TOKENS_LIMIT),
        )


@dataclass(frozen=True)
class OcrJobSpec:
    """Request to read segment crops with Qianfan-OCR.

    Usage:
        Built by the orchestrator, serialised with ``to_payload`` and decoded by
        the OCR worker with ``from_payload``.

    Attributes:
        work_directory: Absolute directory all relative paths refer to.
        backend_name: ``transformers`` (local weights) or ``http`` (OpenAI-compatible server).
        model_directory: Absolute directory of the Qianfan-OCR weights (transformers backend).
        device_name: Torch device for the transformers backend, e.g. ``gpu:0``.
        server_url: Base URL of the OpenAI-compatible server (http backend).
        server_model: Model name exposed by that server (http backend).
        parallel_requests: Concurrent requests, 1..64 (http backend only).
        prompt_text: Instruction sent with every crop.
        ocr_items: Crops to read.
        result_file: Relative path of the JSON-lines result file.
        return_confidences: Ask for ``char_logprobs`` (per-character log-probabilities) in every record.
    """

    work_directory: Path
    backend_name: str
    model_directory: Path | None
    device_name: str
    server_url: str | None
    server_model: str | None
    parallel_requests: int
    prompt_text: str
    ocr_items: tuple[OcrCropItem, ...]
    result_file: str
    return_confidences: bool = False

    def to_payload(self) -> dict:
        """Serialise the job to a JSON-compatible object.

        Usage:
            Call before writing the job file.

        Returns:
            Dictionary with string paths, ready for ``json.dumps``.
        """
        return {
            "work_directory": str(self.work_directory),
            "backend_name": self.backend_name,
            "model_directory": None if self.model_directory is None else str(self.model_directory),
            "device_name": self.device_name,
            "server_url": self.server_url,
            "server_model": self.server_model,
            "parallel_requests": self.parallel_requests,
            "prompt_text": self.prompt_text,
            "ocr_items": [ocr_item.to_payload() for ocr_item in self.ocr_items],
            "result_file": self.result_file,
            "return_confidences": self.return_confidences,
        }

    @classmethod
    def from_payload(cls, job_payload: dict) -> "OcrJobSpec":
        """Decode and validate an OCR job.

        Usage:
            Call in the OCR worker right after reading the job file.

        Args:
            job_payload: Decoded JSON object produced by ``to_payload``.

        Returns:
            The validated OcrJobSpec.

        Raises:
            ValueError: If any field is invalid, or the fields required by the
                chosen backend are missing.
        """
        work_directory = require_absolute_directory(require_string(job_payload, "work_directory"), "work_directory")
        item_entries = require_list(job_payload, "ocr_items", MAX_OCR_ITEMS)
        ocr_request = cls(
            work_directory=work_directory,
            backend_name=require_string(job_payload, "backend_name", 32),
            model_directory=optional_directory(job_payload, "model_directory"),
            device_name=validate_device_name(require_string(job_payload, "device_name", 16)),
            server_url=optional_string(job_payload, "server_url", 2048),
            server_model=optional_string(job_payload, "server_model", 256),
            parallel_requests=require_integer(job_payload, "parallel_requests", 1, MAX_PARALLEL_REQUESTS),
            prompt_text=require_string(job_payload, "prompt_text", MAX_PROMPT_LENGTH),
            ocr_items=tuple(OcrCropItem.from_payload(item_entry, work_directory) for item_entry in item_entries),
            result_file=checked_relative_path(work_directory, require_string(job_payload, "result_file")),
            return_confidences=job_payload.get("return_confidences") is True,
        )
        ocr_request.validate_backend_fields()
        return ocr_request

    def validate_backend_fields(self) -> None:
        """Check that the fields needed by the selected backend are present.

        Usage:
            Called by ``from_payload``; the orchestrator may call it before writing the job.

        Returns:
            None.

        Raises:
            ValueError: If the backend is unknown or its required fields are missing.
        """
        if self.backend_name not in OCR_BACKEND_NAMES:
            raise ValueError(f"Unknown OCR backend {self.backend_name!r}; expected one of {sorted(OCR_BACKEND_NAMES)}.")
        if self.backend_name == TRANSFORMERS_BACKEND and self.model_directory is None:
            raise ValueError("The transformers backend needs 'model_directory'.")
        if self.backend_name == HTTP_BACKEND:
            if self.server_url is None or self.server_model is None:
                raise ValueError("The http backend needs 'server_url' and 'server_model'.")
            validate_http_url(self.server_url)

    def resolve_inside(self, relative_path: str) -> Path:
        """Resolve a relative job path against the working directory.

        Usage:
            Call before opening a crop image or writing the result file.

        Args:
            relative_path: Path relative to ``work_directory``.

        Returns:
            Absolute path inside ``work_directory``.

        Raises:
            ValueError: If the path escapes ``work_directory``.
        """
        return resolve_inside_directory(self.work_directory, relative_path)
